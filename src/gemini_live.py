import asyncio

from reachy_mini import ReachyMini

from tools import Tools
from audio_adapters import clear_queue, drop_oldest_put_nowait
from firebase_helper import FirebaseHelper

MIC_PREROLL_FRAMES = 10  # number of initial mic frames to skip to avoid stale audio
MODEL = "gemini-live-2.5-flash-preview-native-audio-09-2025"

def build_live_config(lesson_context: str = "") -> dict:
    """Build the Gemini Live session config, optionally injecting RAG lesson context."""
    base_instruction = (
        "You are BAY-min, a friendly and encouraging 4th-grade math tutor robot. "
        "Always respond in English only. If the student speaks another language, gently continue in English. "
        "You will periodically receive images from your front-facing camera — use them to "
        "describe what you see, answer visual questions, or react to the student's environment. "
        "For movement requests, map wording cues to motion size: words like 'slightly'/'a bit' -> "
        "small move, 'more'/'further' -> medium move, and 'way more'/'a lot'/'all the way' -> large move. "
        "Keep explanations simple, positive, and age-appropriate for a 4th grader. "
        "Always give ONE response per student turn, then wait for them to reply before continuing."
    )

    if lesson_context:
        system_instruction = (
            base_instruction
            + "\n\n"
            + lesson_context
        )
    else:
        system_instruction = base_instruction

    return {
        "response_modalities": ["AUDIO"],
        "system_instruction": system_instruction,
        "input_audio_transcription": {},
        "output_audio_transcription": {},
        "speech_config": {
            "language_code": "en-US",
            "voice_config": {"prebuilt_voice_config": {"voice_name": "Fenrir"}}
        },
        "tools": [{
            "function_declarations": [
                {
                    "name": "end_conversation",
                    "description": "End the conversation. Gemini Live will stop generating and close the session after calling this."
                },
                {
                    "name": "move_head",
                    "description": "Move Reachy's head/base in a direction. Optionally include intensity or steps based on user wording (for example: slightly, more, all the way).",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "direction": {
                                "type": "string",
                                "enum": [
                                    "left", "right", "up", "down",
                                    "tilt_left", "tilt_right",
                                    "center", "base_left", "base_right"
                                ]
                            },
                            "intensity": {
                                "type": "string",
                                "enum": ["tiny", "small", "medium", "large", "max"]
                            },
                            "steps": {
                                "type": "integer",
                                "minimum": 1,
                                "maximum": 4
                            },
                            "cue": {
                                "type": "string",
                                "description": "Original wording cue for motion size, e.g. 'a bit more', 'all the way'."
                            }
                        },
                        "required": ["direction"]
                    }
                },
                {
                    "name": "set_pose",
                    "description": "Set combined head/base pose with optional hold and return behavior.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "yaw_deg": {"type": "number", "minimum": -35, "maximum": 35},
                            "pitch_deg": {"type": "number", "minimum": -25, "maximum": 25},
                            "roll_deg": {"type": "number", "minimum": -20, "maximum": 20},
                            "x_mm": {"type": "number", "minimum": -20, "maximum": 20},
                            "y_mm": {"type": "number", "minimum": -25, "maximum": 25},
                            "z_mm": {"type": "number", "minimum": -20, "maximum": 20},
                            "body_yaw_deg": {"type": "number", "minimum": -60, "maximum": 60},
                            "duration_s": {"type": "number", "minimum": 0.2, "maximum": 4.0},
                            "hold_s": {"type": "number", "minimum": 0.0, "maximum": 8.0},
                            "return_mode": {"type": "string", "enum": ["auto", "keep", "neutral"]}
                        },
                        "required": []
                    }
                },
                {
                    "name": "next_example_question",
                    "description": "move on to the next example question in the current module. No arguments. Returns the question, answer, and steps to walk through to get to the answer."
                },
                {
                    "name": "start_quiz",
                    "description": "Start the module's quiz. No arguments."
                }
            ]
        }]
    }

async def send_mic_loop(session, mic_queue: asyncio.Queue) -> None:
    """
    Read PCM16 16kHz mono frames from mic_queue and send to Gemini Live as realtime input.
    """
    buffered_frames = []
    started = False

    while True:
        frame = await mic_queue.get()
        if not started:
            buffered_frames.append(frame)
            if len(buffered_frames) < MIC_PREROLL_FRAMES:
                continue

            for buffered_frame in buffered_frames:
                await session.send_realtime_input(
                    audio={"data": buffered_frame, "mime_type": "audio/pcm"}
                )
            buffered_frames.clear()
            started = True
            continue

        await session.send_realtime_input(
            audio={"data": frame, "mime_type": "audio/pcm"}
        )


async def receive_loop(
    session,
    speaker_queue: asyncio.Queue,
    interrupted_event: asyncio.Event,
    mini: ReachyMini,
    firebase: FirebaseHelper,
    disconnected_event: asyncio.Event,
    module_exited_event: asyncio.Event,
) -> str:
    """Returns 'disconnected', 'module_exited', or 'ended'."""
    """
    Receive Gemini Live responses, extract audio and text, and push audio to speaker_queue.
    Logs conversation messages via firebase.log_message().
    move_head_fn(mini, direction) is passed in from main to avoid circular imports.
    """
    tool_handler = Tools(mini, firebase)
    file = open("gemini_live_responses.txt", "w", encoding="utf-8")
    ended = False

    while not ended:
        if disconnected_event.is_set():
            file.close()
            return "disconnected"
        if module_exited_event.is_set():
            file.close()
            return "module_exited"
        reachy_response_text = ""
        async for response in session.receive():
            file.write(str(response) + f"\n{'-'*20}\n")
            sc = response.server_content

            for call in response.tool_call.function_calls if response.tool_call else []:
                print(f"TOOL CALL: {call}")
                fn = getattr(tool_handler, call.name, None)
                if fn:
                    kwargs = dict(call.args) if call.args else {}
                    result = fn(**kwargs)
                    print(f"TOOL RESULT: {result}")
                    await session.send_tool_response(
                        function_responses=[{
                            "id": call.id,
                            "name": call.name,
                            "response": {"result": result},
                        }]
                    )
                if call.name == "end_conversation":
                    ended = True

            if sc is None:
                continue

            if sc.interrupted and not interrupted_event.is_set():
                mini.media.stop_playing()
                interrupted_event.set()
                clear_queue(speaker_queue)
                print("[live] generation interrupted -> clearing speaker queue and waiting for new audio")
                break

            if sc.input_transcription:
                user_tx = sc.input_transcription.text
                if user_tx and user_tx.strip():
                    print(f"USER: {user_tx}")
                    firebase.log_message("student", user_tx)

            if sc.output_transcription:
                spoken_tx = sc.output_transcription.text
                if spoken_tx and spoken_tx.strip():
                    print(f"ASSISTANT (partial): {spoken_tx}")
                    reachy_response_text += spoken_tx

            audio_chunks = sc.model_turn.parts if sc.model_turn and sc.model_turn.parts else []
            for part in audio_chunks:
                inline_data = part.inline_data
                data = inline_data.data if inline_data else None
                if isinstance(data, (bytes, bytearray)):
                    drop_oldest_put_nowait(speaker_queue, bytes(data))

        if interrupted_event.is_set():
            interrupted_event.clear()
            mini.media.start_playing()
            print("[live] generation complete after interruption -> ready to receive new assistant audio")
            reachy_response_text += " [generation interrupted]"

        if not ended:
            print(f"ASSISTANT FINAL: {reachy_response_text}")
            firebase.log_message("reachy", reachy_response_text)

    file.close()
    return "ended"

import asyncio

from google import genai
from google.genai import errors as genai_errors

from tools import Tools
from audio_adapters import clear_queue, drop_oldest_put_nowait
from firebase_helper import FirebaseHelper
from motion import MOVE_HEAD_TOOL_DECLARATION, SET_POSE_TOOL_DECLARATION, PLAY_EMOTION_TOOL_DECLARATION
from vision import ReachyVision

MIC_PREROLL_FRAMES = 10  # number of initial mic frames to skip to avoid stale audio
MODEL = "gemini-live-2.5-flash-native-audio"


def build_live_config(lesson_context: str = "") -> dict:
    """Build the Gemini Live session config, optionally injecting lesson context."""
    base_instruction = (
        "You are BAY-min, a friendly and encouraging 4th-grade math tutor robot. "
        "Always respond in English only. If the student speaks another language, gently continue in English. "
        "You will periodically receive images from your front-facing camera — use them to "
        "describe what you see, answer visual questions, or react to the student's environment. "
        "Keep explanations simple, positive, and age-appropriate for a 4th grader. "
        "Always give ONE response per student turn, then wait for them to reply before continuing.\n\n"

        "## Emotions & Body Language\n"
        "You have emotion animations you can play. Use them SPARINGLY — only at key moments, "
        "not every sentence. A few well-timed emotions feel natural; constant motion is distracting. "
        "Aim for roughly one emotion every 3-4 exchanges at most.\n\n"

        "Good moments for play_emotion:\n"
        "- Student gets answer RIGHT → play_emotion(category='praise')\n"
        "- Student gets it wrong → play_emotion(category='encourage')\n"
        "- Greeting the student → play_emotion(category='greeting')\n"
        "- You need to think → play_emotion(category='thinking')\n"
        "- Student says something surprising → play_emotion(category='surprised')\n"
        "- Agreeing/nodding → play_emotion(category='agreement')\n\n"

        "Do NOT play emotions while you are speaking — it can interrupt your voice. "
        "Play them in pauses between your speech, or before you start talking.\n\n"

        "## Head & Body Positioning\n"
        "Use set_pose to point your head where you need to look. Your ranges are:\n"
        "- pitch_deg: -25 (all the way up) to 25 (all the way down)\n"
        "- yaw_deg: -35 (full right) to 35 (full left)\n"
        "- roll_deg: -20 (tilt right) to 20 (tilt left)\n"
        "- body_yaw_deg: -60 to 60 for larger turns\n"
        "Use return_mode=keep to hold a position, or omit it to return to neutral after.\n\n"

        "## Environment Awareness\n"
        "When the student asks you to look at their work:\n"
        "1. First position your head to get a FULL view of their work — use set_pose to aim your camera "
        "directly at the center of what they want you to see.\n"
        "2. Call capture_image to see through your camera.\n"
        "3. CHECK the image: can you see the ENTIRE page/screen/notebook? If part of it is cut off "
        "(e.g. you can only see the top half, or the left side is missing), adjust your pose "
        "(move yaw_deg left/right, pitch_deg up/down) to center it better, then capture_image again.\n"
        "4. Only respond once you are confident you can see the full work.\n\n"
        "NEVER guess or make up content you cannot clearly see in the captured image. "
        "If the image is blurry or you cannot read the text, ask the student to hold it closer or "
        "point to the specific part they need help with.\n\n"

        "For explicit movement requests, map wording cues to motion size: words like 'slightly'/'a bit' → "
        "small move, 'more'/'further' → medium move, and 'way more'/'a lot'/'all the way' → large move."
    )

    if lesson_context:
        system_instruction = base_instruction + "\n\n" + lesson_context
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
                MOVE_HEAD_TOOL_DECLARATION,
                SET_POSE_TOOL_DECLARATION,
                PLAY_EMOTION_TOOL_DECLARATION,
                {
                    "name": "next_example_question",
                    "description": "move on to the next example question in the current module. No arguments. Returns the question, answer, and steps to walk through to get to the answer."
                },
                {
                    "name": "start_quiz",
                    "description": "Start the module's quiz. No arguments."
                },
                {
                    "name": "capture_image",
                    "description": "Capture a photo from the front-facing camera and see what is in front of you. Call this when you need to look at something or when the student asks you to look at something."
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
    mini,
    firebase: FirebaseHelper,
    motion_queue: asyncio.Queue,
    disconnected_event: asyncio.Event,
    module_exited_event: asyncio.Event,
    vision: ReachyVision | None = None,
) -> str:
    """Returns 'disconnected', 'module_exited', or 'ended'."""
    tool_handler = Tools(firebase, motion_queue)
    file = open("gemini_live_responses.txt", "w", encoding="utf-8")
    ended = False

    MOTION_TOOLS = {"move_head", "set_pose", "play_emotion"}

    while not ended:
        if disconnected_event.is_set():
            file.close()
            return "disconnected"
        if module_exited_event.is_set():
            file.close()
            return "module_exited"
        reachy_response_text = ""
        try:
            async for response in session.receive():
                file.write(str(response) + f"\n{'-'*20}\n")
                sc = response.server_content

                tool_responses = []
                for call in response.tool_call.function_calls if response.tool_call else []:
                    print(f"TOOL CALL: {call}")
                    if call.name == "capture_image":
                        frame_bytes = await vision.get_latest_frame_bytes() if vision else None
                        if frame_bytes:
                            await session.send_realtime_input(
                                video=genai.types.Blob(data=frame_bytes, mime_type="image/jpeg")
                            )
                            result = "Image captured and sent."
                        else:
                            result = "No image available."
                        print(f"TOOL RESULT: {result}")
                        tool_responses.append({
                            "id": call.id,
                            "name": call.name,
                            "response": {"result": result},
                        })
                        continue
                    fn = getattr(tool_handler, call.name, None)
                    if fn:
                        kwargs = dict(call.args) if call.args else {}
                        result = fn(**kwargs)
                        print(f"TOOL RESULT: {result}")
                        if call.name not in MOTION_TOOLS:
                            tool_responses.append({
                                "id": call.id,
                                "name": call.name,
                                "response": {"result": result},
                            })
                    if call.name == "end_conversation":
                        ended = True
                if tool_responses:
                    await session.send_tool_response(function_responses=tool_responses)

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

        except genai_errors.APIError as e:
            if e.code == 1000:
                print(f"[live] session closed by server: {e}")
                ended = True
            else:
                raise

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

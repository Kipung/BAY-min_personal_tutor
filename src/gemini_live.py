import asyncio
import inspect

from google import genai
from google.genai import errors as genai_errors

from tools import Tools
from audio_adapters import clear_queue, drop_oldest_put_nowait
from firebase_helper import FirebaseHelper
from motion import (
    MOVE_HEAD_TOOL_DECLARATION, SET_POSE_TOOL_DECLARATION,
    PLAY_EMOTION_TOOL_DECLARATION, RETURN_HOME_TOOL_DECLARATION,
)
from vision import ReachyVision

MIC_PREROLL_FRAMES = 10  # number of initial mic frames to skip to avoid stale audio
MODEL = "gemini-live-2.5-flash-native-audio"

# capture_image: let queued motion (set_pose/emotion) finish before the snapshot,
# then send multiple frames so Gemini has more than one chance to read the iPad.
CAPTURE_MOTION_SETTLE_S = 1.2
CAPTURE_NUM_FRAMES = 3
CAPTURE_FRAME_SPACING_S = 0.35

_FLOW_DOC = """\
Script:
user: hello baymin
assistant: [call tool to look excited] Hi there! I'm Baymin, today were gonna be learning about [course content]
user: can we start off with an example question?
assistant: [call tool to show an appropriate emotion] call next_example_question() and ask the question it returns
user: is this right?
assistant: [call tool to look down, do not show emotion here] call capture_image() to see the student's work, explain why the student is incorrect
user: can you move on to the next example question?
assistant: [call tool to look excited] call next_example_question() again to get the next question
user: can you start the quiz?
assistant: [call tool to show an appropriate emotion] call start_quiz() to start the quiz, then say good luck!
"""


async def send_flow_context(session, lesson_context: str = "") -> None:
    """Inject operational flow doc + lesson context into conversation history before mic starts."""
    doc = _FLOW_DOC
    if lesson_context:
        doc += "\n## Lesson Context\n" + lesson_context
    await session.send_client_content(
        turns=[{"role": "system", "parts": [{"text": doc}]}],
        turn_complete=False,
    )


def build_live_config() -> dict:
    """Build the Gemini Live session config."""
    system_instruction = (
        "You are BAY-min, a friendly and encouraging 4th-grade math tutor robot. "
        "Always speak English; if the student uses another language, gently continue in English. "
        "Generate exactly ONE spoken response per student message — never start a second audio turn. "
        "Keep all responses warm, age-appropriate, and at most 2 sentences unless asked to explain more."
    )

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
                RETURN_HOME_TOOL_DECLARATION,
                # {
                #     "name": "get_face_position",
                #     "description": (
                #         "Detect where the student's face is in your camera view. "
                #         "Returns a description of their position (left/right/above/below center) "
                #         "and suggested yaw_deg/pitch_deg adjustments to center on them. "
                #         "Use this at the start of a session, after large turns, or before capturing work."
                #     ),
                # },
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
    tool_handler = Tools(firebase, motion_queue, vision=vision)
    file = open("gemini_live_responses.txt", "w", encoding="utf-8")
    ended = False

    async def _process_responses():
        nonlocal ended
        # After producing a response, require new user speech before allowing the next one.
        # This suppresses spontaneous follow-up turns the model sometimes emits after tool use.
        require_user_input = False

        while not ended:
            reachy_response_text = ""
            user_spoke = False
            try:
                async for response in session.receive():
                    file.write(str(response) + f"\n{'-'*20}\n")
                    sc = response.server_content

                    tool_responses = []
                    for call in response.tool_call.function_calls if response.tool_call else []:
                        print(f"TOOL CALL: {call}")
                        if call.name == "capture_image":
                            # Let any queued set_pose / play_emotion settle before grabbing frames.
                            await asyncio.sleep(CAPTURE_MOTION_SETTLE_S)

                            frames_sent = 0
                            for i in range(CAPTURE_NUM_FRAMES):
                                frame_bytes = await vision.get_latest_frame_bytes() if vision else None
                                if frame_bytes:
                                    await session.send_realtime_input(
                                        video=genai.types.Blob(data=frame_bytes, mime_type="image/jpeg")
                                    )
                                    frames_sent += 1
                                if i < CAPTURE_NUM_FRAMES - 1:
                                    await asyncio.sleep(CAPTURE_FRAME_SPACING_S)

                            if frames_sent > 0:
                                result = f"OK — {frames_sent} frames sent."
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
                            if inspect.iscoroutinefunction(fn):
                                result = await fn(**kwargs)
                            else:
                                result = fn(**kwargs)
                            print(f"TOOL RESULT: {result}")
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

                    # if sc.interrupted and not interrupted_event.is_set():
                    #     mini.media.stop_playing()
                    #     interrupted_event.set()
                    #     clear_queue(speaker_queue)
                    #     print("[live] generation interrupted -> clearing speaker queue and waiting for new audio")
                    #     break

                    if sc.input_transcription:
                        user_tx = sc.input_transcription.text
                        if user_tx and user_tx.strip():
                            user_spoke = True
                            require_user_input = False
                            print(f"USER: {user_tx}")
                            firebase.log_message("student", user_tx)

                    if require_user_input:
                        continue

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
                if reachy_response_text:
                    print(f"ASSISTANT FINAL: {reachy_response_text}")
                    firebase.log_message("reachy", reachy_response_text)
                    require_user_input = True
                elif not user_spoke:
                    print("[live] suppressed spontaneous model turn (no user input received)")

    async def _watch_exit_events():
        # Returns as soon as either exit event fires, unblocking asyncio.wait.
        dc = asyncio.create_task(disconnected_event.wait())
        me = asyncio.create_task(module_exited_event.wait())
        try:
            await asyncio.wait([dc, me], return_when=asyncio.FIRST_COMPLETED)
        finally:
            dc.cancel()
            me.cancel()

    process_task = asyncio.create_task(_process_responses())
    watch_task = asyncio.create_task(_watch_exit_events())

    try:
        await asyncio.wait(
            [process_task, watch_task],
            return_when=asyncio.FIRST_COMPLETED,
        )
    finally:
        for task in [process_task, watch_task]:
            if not task.done():
                task.cancel()
                try:
                    await task
                except (asyncio.CancelledError, Exception):
                    pass
        file.close()

    if disconnected_event.is_set():
        return "disconnected"
    if module_exited_event.is_set():
        return "module_exited"
    return "ended"

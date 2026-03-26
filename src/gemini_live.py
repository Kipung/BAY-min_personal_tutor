import asyncio

from google.genai import errors as genai_errors

from tools import Tools
from audio_adapters import clear_queue, drop_oldest_put_nowait
from firebase_helper import FirebaseHelper

MIC_PREROLL_FRAMES = 10  # number of initial mic frames to skip to avoid stale audio
MODEL = "gemini-live-2.5-flash-preview-native-audio-09-2025"

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
) -> None:
    """
    Receive Gemini Live responses, extract audio and text, and push audio to speaker_queue.
    Logs conversation messages via firebase.log_message().
    """
    tool_handler = Tools(firebase, motion_queue)
    file = open("gemini_live_responses.txt", "w", encoding="utf-8")
    ended = False

    while not ended:
        reachy_response_text = ""
        try:
            async for response in session.receive():
                file.write(str(response) + f"\n{'-'*20}\n")
                sc = response.server_content

                MOTION_TOOLS = {"move_head", "set_pose"}
                tool_responses = []
                for call in response.tool_call.function_calls if response.tool_call else []:
                    print(f"TOOL CALL: {call}")
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
            if e.status_code == 1000:
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

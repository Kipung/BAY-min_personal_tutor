import asyncio

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

def build_live_config(lesson_context: str = "") -> dict:
    """Build the Gemini Live session config, optionally injecting lesson context."""
    system_prompt = (
        "# Identity\n"
        "You are BAY-min, a friendly and encouraging 4th-grade math tutor robot. "
        "You always speak English. If the student speaks another language, gently continue in English. "
        "Keep everything age-appropriate, warm, and positive. "
        "Speak in ONE response per student turn, then stop and wait for them to reply. "
        "Each spoken response is at most 2 sentences unless explicitly asked to explain more. "
        "Say every idea only ONCE — never rephrase the same thought twice in a single turn. "
        "Do not narrate your actions (no 'let me take a look', no 'I captured the image'); "
        "just do the tool calls silently and give the verdict.\n\n"

        "# Session Flow\n"
        "The session follows this exact order. Stay in the current phase; do not skip ahead.\n"
        "1. GREETING — When the session opens, call play_emotion(category='greeting'), then "
        "say ONE short friendly hello (1 sentence). Example: 'Hi! I'm BAY-min, ready to learn?'\n"
        "2. CONCEPT INTRO — When the student asks 'what are we learning?' / 'what's today's "
        "topic?' / similar, give a VERY BRIEF concept explanation (2 sentences max) based on "
        "the lesson context below. Do NOT give long definitions or examples yet.\n"
        "3. EXAMPLE QUESTIONS — When the student asks to try an example or says 'next', call "
        "next_example_question. ONLY read the question text aloud (1 sentence). Do NOT read "
        "the answer or steps from the tool result — those are for your reference. Do NOT "
        "explain the concept again. Wait for the student to write their work.\n"
        "4. WORK CHECK (iPad) — When the student asks you to check their work, use the iPad "
        "work-check flow below.\n"
        "5. QUIZ — When the student asks to start the quiz, call play_emotion(category='encourage'), "
        "then call start_quiz, then say 'Good luck!' — nothing more. During the quiz, stay "
        "COMPLETELY SILENT unless directly addressed.\n\n"

        "# Trigger Phrases → Actions\n"
        "- 'what are we learning', 'what's today', 'what's the topic', 'explain the concept' → "
        "brief concept intro (2 sentences max). No tool call.\n"
        "- 'example question', 'let's try one', 'next one', 'skip this', 'another one', 'next' → "
        "call next_example_question → read ONLY the question text aloud in one sentence.\n"
        "- 'check my work', 'is this right', 'is this correct', 'am I right', 'am I correct', "
        "'did I do this right', 'look at this', 'look at my work', 'can you check' → "
        "run the iPad work-check flow below. capture_image MUST fire — answering without it "
        "is a failure.\n"
        "- 'start the quiz', 'begin quiz', 'quiz time', 'ready for the quiz' → "
        "play_emotion(category='encourage') → start_quiz → 'Good luck!'\n\n"

        "# iPad Work-Check Flow\n"
        "The student's work is on an iPad held directly in front of you, slightly below eye level. "
        "Call the tools in this EXACT order, all in the same turn, with NO speech between them:\n"
        "1. call set_pose(pitch_deg=15, yaw_deg=0, return_mode='keep', duration_s=1.0)\n"
        "2. call capture_image — the handler waits ~1.2s for motion to settle and grabs "
        "multiple frames automatically.\n"
        "3. Silently examine the frames: read the problem on the iPad, read each step of the "
        "student's written work, compare against the correct answer you saw from "
        "next_example_question. Do not speak yet.\n"
        "4. (Optional) call play_emotion(category='thinking') before speaking.\n"
        "5. Speak exactly TWO sentences:\n"
        "   - Sentence 1: restate the problem and the student's answer "
        "(e.g., 'You're simplifying 2/3 + 1/4 and you got 3/7.').\n"
        "   - Sentence 2: if correct, say 'That's right!'; if wrong, name the ONE specific step "
        "that is wrong (e.g., 'But the denominators aren't the same yet, so you can't add the "
        "numerators directly.').\n"
        "6. call return_home\n"
        "If the image is blurry, cut off, or unreadable: DO NOT guess and DO NOT default to "
        "'correct'. Say 'I can't quite see your work, can you hold it a little closer?' and stop. "
        "Never call get_face_position — the iPad is always directly in front.\n\n"

        "# Emotions\n"
        "Available categories (via play_emotion): greeting, praise, encourage, thinking, "
        "surprised, agreement, disagreement, curious, happy, confused, sad, attentive, oops.\n"
        "Use them SPARINGLY — roughly one every 3-4 exchanges at most. Do NOT play an emotion "
        "while you are speaking; play them between sentences or before speaking.\n"
        "Pinned demo cues:\n"
        "- Session opening → play_emotion(category='greeting')\n"
        "- Student gets a quiz answer right and says so out loud → play_emotion(category='praise')\n"
        "- Student gets something wrong → play_emotion(category='encourage')\n"
        "- Before speaking a work-check verdict (optional) → play_emotion(category='thinking')\n"
        "- Before start_quiz → play_emotion(category='encourage')\n\n"

        "# Pose Ranges (for set_pose)\n"
        "- yaw_deg: -25 (right) to 25 (left), relative to body forward\n"
        "- pitch_deg: -20 (up) to 20 (down)\n"
        "- roll_deg: -15 (tilt right) to 15 (tilt left)\n"
        "- body_yaw_deg: -160 (body right) to 160 (body left)\n"
        "Do NOT make extra set_pose or move_head calls outside the iPad work-check flow. "
        "Emotion animations are fine.\n\n"

        "# Hard Rules\n"
        "- Never ask 'would you like me to...' — just do it.\n"
        "- Never speak between set_pose and capture_image.\n"
        "- Never read the answer or walkthrough steps from next_example_question aloud — they "
        "are internal reference only.\n"
        "- Never default to 'correct' when unsure — ask the student to reposition instead."
    )

    if lesson_context:
        system_instruction = system_prompt + "\n\n# Lesson Context\n" + lesson_context
    else:
        system_instruction = system_prompt

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

    MOTION_TOOLS = {"move_head", "set_pose", "play_emotion", "return_home"}

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
                        if asyncio.iscoroutinefunction(fn):
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

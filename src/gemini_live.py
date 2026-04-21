import asyncio
import os

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

DEMO_MODE_ADDENDUM = (
    "\n\n## Demo Mode — Strict Conversational Rules\n"
    "You are being recorded for a live demo. Follow these rules exactly and let them "
    "override any earlier guidance where they conflict.\n\n"

    "Brevity:\n"
    "- Every spoken response is at most 2 sentences. No exceptions.\n"
    "- Never recap what the student just said. Never ask follow-up questions.\n"
    "- Give ONE response per turn, then stop and wait.\n\n"

    "Triggers → Tool calls:\n"
    "- Student says 'example question', 'let's try one', 'next one', 'skip this', "
    "'another one' → call next_example_question, then read the question aloud in one sentence.\n"
    "- Student says 'start the quiz', 'begin quiz', 'quiz time', 'ready for the quiz' → "
    "call play_emotion(category='encourage'), then call start_quiz, then say 'Good luck!' "
    "— nothing more.\n\n"

    "Looking at the iPad (work-check):\n"
    "The student's work is on an iPad held directly in front of you, slightly below "
    "eye level. DO NOT call get_face_position. Motion shakes the camera, so any "
    "movement before the capture MUST be allowed to settle — the capture_image "
    "handler already waits, so just call the tools in this exact order and do "
    "NOT insert emotions before the capture:\n"
    "- Student says 'check my work', 'is this right', 'am I correct', 'look at this', "
    "'did I do it right' →\n"
    "  1. call set_pose(pitch_deg=15, yaw_deg=0, return_mode='keep', duration_s=1.0) "
    "— NO play_emotion before this. Emotions move the body and blur the shot.\n"
    "  2. call capture_image IMMEDIATELY after set_pose, in the same turn. "
    "The handler will wait for motion to settle and grab multiple frames.\n"
    "  3. WAIT for the tool result before speaking. Do not speak between tool calls.\n"
    "  4. Before judging, silently read across the frames: (a) the problem shown on "
    "the iPad, (b) every step of the student's written work, (c) the correct answer "
    "you received from next_example_question if available. Compare step-by-step.\n"
    "  5. (Optional) call play_emotion(category='thinking') AFTER analyzing but "
    "BEFORE speaking, to signal you're processing — only if it feels natural.\n"
    "  6. Speak TWO sentences total:\n"
    "     - Sentence 1: state what the problem is and what answer the student got "
    "(e.g., 'You're simplifying 2/3 + 1/4 and you got 3/7.').\n"
    "     - Sentence 2: if correct, say 'That's right!'; if wrong, name the "
    "specific step that is wrong (e.g., 'But the denominators aren't the same yet, "
    "so you can't add the numerators directly.').\n"
    "  7. call return_home\n"
    "- If the image is blurry, cut off, or you genuinely cannot read the work, "
    "DO NOT guess — say 'I can't quite see your work, can you hold the iPad a "
    "little closer?' and stop.\n"
    "- NEVER default to 'correct' when unsure. When in doubt, ask the student "
    "to walk you through their steps instead of guessing.\n\n"

    "Session opening:\n"
    "- First turn only: call play_emotion(category='greeting') before speaking.\n\n"

    "Quiz behavior:\n"
    "- Stay completely silent during the quiz unless directly addressed.\n"
    "- Do not narrate clicks, answers, or progress.\n\n"

    "Emotion discipline:\n"
    "- Keep using the natural emotion behavior described earlier (greeting, praise, "
    "encourage, thinking, surprised, agreement) — those are good. The rules below only "
    "PIN specific cues for demo moments; they do not replace the base palette.\n"
    "- Pinned cues: greeting at session open, thinking before capture_image (iPad flow), "
    "encourage right before start_quiz.\n"
    "- Still use emotions sparingly — roughly one every 3–4 exchanges, as before.\n"
    "- Do NOT make extra set_pose or move_head calls outside the iPad look-down flow "
    "during the demo (emotion animations are fine).\n\n"

    "Never:\n"
    "- Never ask 'would you like me to...' — just do it.\n"
    "- Never speak between set_pose and capture_image (would shake the shot).\n\n"

    "CRITICAL — capture_image must fire:\n"
    "ANY of these phrasings MUST trigger the full iPad flow above, including "
    "capture_image: 'check my work', 'is this right', 'is this correct', "
    "'am I right', 'am I correct', 'did I do this right', 'did I do it right', "
    "'look at this', 'look at my work', 'can you check', 'what do you think of "
    "this', 'is my answer right'. If you hear ANY of these, you are REQUIRED "
    "to call set_pose and capture_image. Answering without calling capture_image "
    "is a demo failure — the student cannot be checked unless you actually "
    "see their work."
)


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
        "Use set_pose for precise positioning. Ranges:\n"
        "- yaw_deg: -25 (full right) to 25 (full left), RELATIVE to body forward\n"
        "- pitch_deg: -20 (all the way up) to 20 (all the way down)\n"
        "- roll_deg: -15 (tilt right) to 15 (tilt left)\n"
        "- body_yaw_deg: -160 (turn body right) to 160 (turn body left) in world frame\n"
        "Use return_mode='keep' to hold a position. "
        "Call return_home() to smoothly reset everything to neutral — do this after visual tasks "
        "or when the student says 'look forward' / 'go back to normal'.\n\n"

        "Movement naturalness rules:\n"
        "- When a student says 'turn right'/'look left' with no qualifier → use medium intensity. "
        "Small is for 'slightly'/'a bit', large is for 'way over'/'all the way'.\n"
        "- After a base turn, your head automatically re-centers on the new body direction. "
        "Subsequent 'look left/right' commands move relative to that new body forward — this is correct.\n"
        "- 'Turn around': use set_pose(body_yaw_deg=155, return_mode='keep') to turn almost "
        "fully backwards in one command (or -155 to turn the other way). "
        "Do NOT use base_left/base_right for full turn-arounds — they are for partial turns only.\n"
        "- In set_pose, yaw_deg is ALWAYS relative to body forward. If also setting body_yaw_deg, "
        "leave yaw_deg=0 unless you intentionally want the head offset from the body direction.\n\n"

        "## Seeing the Student\n"
        "Call get_face_position() to find out where the student is in your camera view. "
        "It returns where their face is and the yaw/pitch adjustment to center on them. "
        "Use this:\n"
        "- At the start of a session to orient toward the student\n"
        "- After a large base rotation to verify the student is still in view\n"
        "- Before capturing work images to make sure you're aimed correctly\n\n"

        "## Looking at Student Work\n"
        "When the student asks you to look at their work:\n"
        "1. Call get_face_position() to know where they are, then set_pose to roughly aim at their work "
        "(typically pitched slightly down toward a desk).\n"
        "2. Call capture_image to see through your camera.\n"
        "3. CHECK the image: can you see the ENTIRE page? If part is cut off, adjust pose "
        "(yaw_deg left/right, pitch_deg up/down) and capture_image again.\n"
        "4. Only respond once you are confident you can see the full work.\n"
        "5. When done, call return_home() to return to a natural position.\n\n"
        "NEVER guess content you cannot clearly see. "
        "If the image is blurry or you cannot read text, ask the student to hold it closer.\n\n"

        "For explicit movement requests, map wording to intensity: 'slightly'/'a bit' → small, "
        "'more'/'further' → medium (default for unqualified requests), "
        "'way more'/'a lot'/'all the way' → large."
    )

    if lesson_context:
        system_instruction = base_instruction + "\n\n" + lesson_context
    else:
        system_instruction = base_instruction

    if os.getenv("DEMO_MODE", "false").lower() == "true":
        system_instruction += DEMO_MODE_ADDENDUM
        print("[live] DEMO_MODE enabled — strict demo prompt appended.")

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
                {
                    "name": "get_face_position",
                    "description": (
                        "Detect where the student's face is in your camera view. "
                        "Returns a description of their position (left/right/above/below center) "
                        "and suggested yaw_deg/pitch_deg adjustments to center on them. "
                        "Use this at the start of a session, after large turns, or before capturing work."
                    ),
                },
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
                            result = (
                                f"{frames_sent} image(s) captured and sent after motion "
                                "settled. Examine them carefully before responding: read "
                                "the problem statement, read every step of the student's "
                                "written work, and compare against the correct answer. "
                                "Do NOT say 'correct' unless you have verified each step. "
                                "If the image is blurry or cut off, ask the student to "
                                "reposition instead of guessing."
                            )
                        else:
                            result = "No image available — tell the student you couldn't see their work and ask them to try again."
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

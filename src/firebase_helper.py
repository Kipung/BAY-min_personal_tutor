import asyncio
from datetime import datetime

import firebase_admin
from firebase_admin import credentials, firestore


class FirebaseHelper:
    """
    Manages Firebase connection and the current active user/module session.
    Call set_loop() once at startup, then set_user() after receiving the active user.
    """

    def __init__(self):
        if not firebase_admin._apps:
            creds = credentials.Certificate("credentials.json")
            firebase_admin.initialize_app(creds)
        self.db = firestore.client()
        self.user_id: str = None
        self.user_doc_ref = None
        self.module_id: str = None
        self._profile_watch = None
        self._reachy_watch = None
        self._loop: asyncio.AbstractEventLoop = None
        self.module_selected_event: asyncio.Event = None
        self.module_exited_event: asyncio.Event = None

    def set_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        """Store the running event loop. Call once at startup."""
        self._loop = loop

    def set_user(self, user_id: str) -> None:
        """Store the active user so Firestore operations can target the right document."""
        self.user_id = user_id
        self.user_doc_ref = self.db.collection("user_profiles").document(user_id)

    def stop(self) -> None:
        """Unsubscribe all Firestore listeners."""
        if self._profile_watch:
            self._profile_watch.unsubscribe()
            self._profile_watch = None
        if self._reachy_watch:
            self._reachy_watch.unsubscribe()
            self._reachy_watch = None

    def reset(self) -> None:
        """Stop listeners and clear user state — call before re-entering State 1."""
        self.stop()
        self.user_id = None
        self.user_doc_ref = None
        self.module_id = None

    def log_message(self, sender: str, message: str) -> None:
        """
        Log a conversation message to user_profiles/{user_id}/modules/{module_id}/messages.
        sender should be 'student', 'reachy', or 'system'.
        """
        if not self.user_doc_ref or not self.module_id:
            print(f"[firebase] log_message skipped (no active module): [{sender}] {message[:60]}")
            return
        self.user_doc_ref \
            .collection("modules").document(self.module_id) \
            .collection("messages") \
            .add({"from": sender, "message": message, "createdAt": datetime.now()})

    def get_next_example_question(self) -> str:
        if not self.user_doc_ref or not self.module_id:
            return "No active module selected. Please select a module first."
        next_num = self.user_doc_ref.collection("modules").document(self.module_id).get().to_dict().get("example_question_num", 0) + 1

        module_data = self.db.collection("modules").document(self.module_id).get().to_dict() or {}
        example_questions = (
            module_data.get("quiz_questions", {}).get("guided", [])
        )
        if next_num >= len(example_questions):
            return "There are no more example questions, move on to the quiz."

        self.user_doc_ref.collection("modules").document(self.module_id).set({"example_question_num": next_num}, merge=True)

        prefix = "FINAL EXAMPLE QUESTION\n" if next_num == len(example_questions) - 1 else ""
        return prefix + str(example_questions[next_num])

    def get_lesson_data(self) -> str:
        if not self.user_doc_ref or not self.module_id:
            raise RuntimeError("set_user() must be called and a module must be selected before get_lesson_data()")
        module_data = self.db.collection("modules").document(self.module_id).get().to_dict() or {}
        try:
            concepts = [
                {"term": c.get("term", ""), "definition": c.get("definition", "").strip()}
                for c in module_data.get("concepts", [])
            ]
        except Exception as e:
            try:
                concepts = ', '.join(module_data.get("concepts", []))
            except Exception as e:
                concepts = ""
        lesson = {
            "title": module_data.get("title", ""),
            "essential_question": module_data.get("essential_question", "").strip(),
            "concepts": concepts,
        }
        return str(lesson)
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
        self.disconnected_event: asyncio.Event = None

    def set_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        """Store the running event loop and initialise async events. Call once at startup."""
        self._loop = loop
        self.module_selected_event = asyncio.Event()
        self.module_exited_event = asyncio.Event()
        self.disconnected_event = asyncio.Event()

    def set_user(self, user_id: str) -> None:
        """
        Store the active user and start listeners so that:
        - module_selected_event is set when active_module_id becomes non-null
        - module_exited_event is set when active_module_id becomes null
        - disconnected_event is set when active_user in reachy-mini/reachy1 becomes null
        """
        self.user_id = user_id
        self.user_doc_ref = self.db.collection("user_profiles").document(user_id)

        # Reset events for this new session
        self.module_selected_event.clear()
        self.module_exited_event.clear()
        self.disconnected_event.clear()

        # Read initial module_id synchronously to avoid a race with the listener
        profile = self.user_doc_ref.get()
        if profile.exists:
            self.module_id = profile.to_dict().get("active_module_id")
            print(f"[firebase] Initial module: {self.module_id}")

        def on_profile_update(snapshots, _changes, _read_time):
            for snap in snapshots:
                new_module = snap.to_dict().get("active_module_id")
                if new_module == self.module_id:
                    return
                self.module_id = new_module
                print(f"[firebase] Module changed: {self.module_id}")
                if new_module:
                    self._loop.call_soon_threadsafe(self.module_selected_event.set)
                else:
                    self._loop.call_soon_threadsafe(self.module_exited_event.set)

        self._profile_watch = self.user_doc_ref.on_snapshot(on_profile_update)

        # Watch reachy-mini/reachy1 for disconnect (active_user → null)
        reachy_ref = self.db.collection("reachy-mini").document("reachy1")

        def on_reachy_update(snapshots, _changes, _read_time):
            for snap in snapshots:
                active_user = snap.to_dict().get("active_user")
                if not active_user:
                    print("[firebase] User disconnected (active_user cleared).")
                    self._loop.call_soon_threadsafe(self.disconnected_event.set)

        self._reachy_watch = reachy_ref.on_snapshot(on_reachy_update)

    def stop(self) -> None:
        """Unsubscribe all Firestore listeners."""
        if self._profile_watch:
            self._profile_watch.unsubscribe()
            self._profile_watch = None
        if self._reachy_watch:
            self._reachy_watch.unsubscribe()
            self._reachy_watch = None

    def reset(self) -> None:
        """Stop listeners, clear user/module state, and reset events — call before re-entering State 1."""
        self.stop()
        self.user_id = None
        self.user_doc_ref = None
        self.module_id = None
        self.module_selected_event.clear()
        self.module_exited_event.clear()
        self.disconnected_event.clear()

    def log_message(self, sender: str, message: str) -> None:
        """
        Log a conversation message to user_profiles/{user_id}/modules/{module_id}/messages.
        sender should be 'student', 'reachy', or 'system'.
        """
        if not self.user_doc_ref or not self.module_id:
            raise RuntimeError("set_user() must be called and user must select a module before log_message()")
        self.user_doc_ref \
            .collection("modules").document(self.module_id) \
            .collection("messages") \
            .add({"from": sender, "message": message, "createdAt": datetime.now()})

    def get_next_example_question(self) -> str:
        if not self.user_doc_ref or not self.module_id:
            raise RuntimeError("set_user() must be called and user must select a module before get_next_example_question()")
        next_num = self.user_doc_ref.collection("modules").document(self.module_id).get().to_dict().get("example_question_num") + 1

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
            raise RuntimeError("set_user() must be called and user must select a module before get_lesson_context()")
        module_data = self.db.collection("modules").document(self.module_id).get().to_dict() or {}
        #dont attach stuff under quiz_questions
        module_data.pop("quiz_questions", None)
        return str(module_data)
from datetime import datetime

import firebase_admin
from firebase_admin import credentials, firestore


class FirebaseHelper:
    """
    Manages Firebase connection and the current active user/module session.
    Call set_user() after receiving the active user from Bluetooth.
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

    def set_user(self, user_id: str) -> None:
        """
        Store the active user and start a listener on their profile so that
        module_id stays current whenever active_module_id changes in Firestore.
        """
        self.user_id = user_id
        self.user_doc_ref = self.db.collection("user_profiles").document(user_id)

        def on_profile_update(snapshots, _changes, _read_time):
            for snap in snapshots:
                self.module_id = snap.to_dict().get("active_module_id")
                print(f"[firebase] User switched to module: {self.module_id}")

        self._profile_watch = (
            self.user_doc_ref.on_snapshot(on_profile_update)
        )

    def stop(self) -> None:
        """Unsubscribe the profile listener. Call on shutdown."""
        if self._profile_watch:
            self._profile_watch.unsubscribe()

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
            module_data.get("instructional_content", {}).get("example_walkthrough", [])
        )
        if next_num >= len(example_questions):
            return "There are no more example questions, move on to the quiz."

        self.user_doc_ref.collection("modules").document(self.module_id).set({"example_question_num": next_num}, merge=True)

        prefix = "FINAL EXAMPLE QUESTION\n" if next_num == len(example_questions) - 1 else ""
        return prefix + str(example_questions[next_num])
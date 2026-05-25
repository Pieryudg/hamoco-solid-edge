class MouseAdapter:

    def __init__(self, controller):
        self.controller = controller

    def operate(self, hand, palm_center, confidence, min_confidence):
        self.controller.operate_mouse(hand, palm_center, confidence, min_confidence=min_confidence)

    def release_controls(self):
        self.controller.release_controls()

# Exceptions
class InferImageError(Exception):
    pass

class ProcessImageError(InferImageError):
    pass

class OCRRunError(ProcessImageError):
    pass

class ParseDialogueError(Exception):
    pass

class PaddleAugmentationError(Exception):
    pass

class SaveJSONError(Exception):
    pass
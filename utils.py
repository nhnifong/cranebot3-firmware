import asyncio
import functools

def motion_task(func):
    """
    A decorator that enforces a coroutine is run as a motion task.
    
    It ensures that 'invoke_motion_task' exists somewhere in the call stack,
    preventing direct calls that bypass the task management system, while still
    allowing motion tasks to call other motion tasks directly.
    """
    @functools.wraps(func)
    async def wrapper(self, *args, **kwargs):
        # Check if 'invoke_motion_task' appears anywhere in the call stack.
        is_managed = any(frame.function == 'invoke_motion_task' for frame in inspect.stack())
        
        if not is_managed:
            raise RuntimeError(
                f"Motion task '{func.__name__}' was not started by a managed process. "
                f"The call chain must originate from 'invoke_motion_task'."
            )
        
        # If the check passes, execute the original coroutine.
        return await func(self, *args, **kwargs)
    return wrapper


def constrain(value, minimum, maximum):
    return max(minimum, min(value, maximum))
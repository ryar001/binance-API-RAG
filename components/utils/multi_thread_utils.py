import threading
from typing import Callable, Any, List, Tuple, Dict
import functools

class MultiThreadUtils:
    def __init__(self):
        """
        Initializes the `MultiThreadUtils` object.

        This method creates an empty dictionary `thread_pool` to store threads and a lock `lock` to synchronize access to the dictionary.

        Parameters:
            None

        Returns:
            None
        """
        self.thread_pool: Dict[str, threading.Thread] = {}
        self.lock = threading.Lock()


    def thread_lock(self,func,*args, **kwargs):
        """
        Decorator that locks any function it wraps using a lock.

        :param func: The function to be wrapped.
        :type func: Callable
        :return: The wrapped function.
        :rtype: Callable
        """
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            """
            Wrapper function that locks the wrapped function using a lock.

            :param args: The positional arguments to pass to the wrapped function.
            :type args: Tuple
            :param kwargs: The keyword arguments to pass to the wrapped function.
            :type kwargs: Dict
            :return: The result of the wrapped function.
            """
            with self.lock:
                return func(*args, **kwargs)
        return wrapper(*args, **kwargs)

    def create_thread(self, name: str, target: Callable, args: Tuple = None, kwargs: dict = None) -> threading.Thread:
        """
        Creates a thread with a given name and adds it to the thread dictionary.

        :param name: The name of the thread.
        :type name: str
        :param target: The target function to run in the thread.
        :type target: Callable
        :param args: The arguments to pass to the target function. Defaults to None.
        :type args: Tuple, optional
        :param kwargs: The keyword arguments to pass to the target function. Defaults to None.
        :type kwargs: dict, optional
        :raises ValueError: If a thread with the same name already exists.
        :return: The created thread.
        :rtype: threading.Thread
        """
  
        if name in self.thread_pool:
            raise ValueError(f"A thread with the name '{name}' already exists.")

        def wrapper(*args, **kwargs):
            # storing the result
            result = target(*args, **kwargs)
            thread.result = result

        thread = threading.Thread(target=wrapper, args=args, kwargs=kwargs, name=name)
        self.thread_pool[name] = thread
        return thread

    def start_thread(self, name: str):
        """Starts the thread with the given name."""
        thread = self.thread_pool.get(name)
        if thread:
            thread.start()
        else:
            raise ValueError(f"No thread found with the name '{name}'.")

    def create_and_start_thread(self, name: str, target: Callable, args: Tuple = None, kwargs: dict = None):
        """Creates and starts a new thread with the given name."""
        thread = self.create_thread(name, target, args, kwargs)
        self.start_thread(name)
        return thread

    def join_thread(self, name: str):
        """Waits for the thread with the given name to finish."""
        thread = self.thread_pool.get(name)
        if thread:
            thread.join()
        else:
            raise ValueError(f"No thread found with the name '{name}'.")

    def start_all(self):
        """Starts all threads."""
        for thread in self.thread_pool.values():
            thread.start()

    def join_all(self):
        """Waits for all threads to finish."""
        for thread in self.thread_pool.values():
            thread.join()

    def clear_threads(self):
        """Clears the list of threads."""
        self.thread_pool.clear()

    def get_thread_result(self, name: str) -> Any:
        """Returns the result of the thread with the given name."""
        thread = self.thread_pool.get(name)
        if thread:
            return thread.result
        else:
            raise ValueError(f"No thread found with the name '{name}'.")
        
    def get_all_thread_results(self) -> Dict[str,Any]:
        """Returns a list of tuples containing the name and result of all threads."""
        return {i:j.result for i,j in self.thread_pool.items()}

# Example usage
if __name__ == "__main__":
    import time

    def print_numbers():
        for i in range(5):
            print(i)
            time.sleep(1)

    def print_letters():
        for letter in 'abcde':
            print(letter)
            time.sleep(1)

    mt_utils = MultiThreadUtils()
    mt_utils.create_thread(name="numbers", target=print_numbers)
    mt_utils.create_thread(name="letters", target=print_letters)
    
    mt_utils.start_all()
    mt_utils.join_all()
    mt_utils.clear_threads()

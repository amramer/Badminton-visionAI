import logging
from time import time
from typing import List, Optional, Callable
import sys

class ProgressTracker:
    """Progress tracking system for multi-step processes with execution control"""

    def __init__(self, total_steps: int, logger: Optional[logging.Logger] = None):
        self.total_steps = total_steps
        self.current_step = 0
        self.step_times = []
        self.step_names = []
        self.start_time = time()
        self.logger = logger
        self.max_steps = None
        self._execution_stopped = False

    STEP_SYMBOLS = {
        'Application Initialization': '🚀',
        'Configuration Loading': '⚙️',
        'Video Processing': '🎬',
        'Court Detection': '🏟️',
        'Player Tracking': '👥',
        'Ball Tracking': '🎾',
        'Side Court Visualization': '🏸',
        'Dashboard Generation': '📊',
        'Finalization': '🏆'
    }

    def set_max_steps(self, max_steps: Optional[int]):
        """Set maximum number of steps to execute"""
        self.max_steps = max_steps

    def _check_max_steps(self):
        """Internal method to check if max steps reached"""
        if self._execution_stopped:
            return False
        if self.max_steps is not None and self.current_step >= self.max_steps:
            self._execution_stopped = True
            if self.logger:
                self.logger.info(f"⏹️ Max steps ({self.max_steps}) reached - stopping execution")
            self.summary()
            sys.exit(0)  # Clean exit when max steps reached
        return True

    def begin_step(self, step_name: str) -> None:
        """Start a new processing step with execution control"""
        if not self._check_max_steps():
            return
            
        self.current_step += 1
        self.step_names.append(step_name)
        symbol = self.STEP_SYMBOLS.get(step_name, '➡️')
        step_header = (
            f"\n{'='*50}\n"
            f"{symbol} STEP {self.current_step}/{self.total_steps}: {step_name.upper()}\n"
            f"{'='*50}"
        )
        # print(step_header)
        if self.logger:
            self.logger.info(step_header)
        self.step_start = time()

    def end_step(self, success: bool = True, message: Optional[str] = None) -> None:
        """Complete the current step"""
        if self._execution_stopped:
            return
            
        step_time = time() - self.step_start
        self.step_times.append(step_time)

        status = "✅ COMPLETED" if success else "❌ FAILED"
        step_footer = (
            f"\n{'='*20} STEP {status} IN {step_time:.2f}s {'='*20}"
        )
        print(step_footer)
        if self.logger:
            self.logger.info(step_footer)

        if message:
            print(f"NOTE: {message}")
            if self.logger:
                self.logger.info(f"NOTE: {message}")

    def summary(self) -> None:
        """Print processing summary"""
        total_time = time() - self.start_time
        summary = [f"\n{'='*50} 📊 PROCESSING SUMMARY\n{'='*50}"]

        executed_steps = min(len(self.step_times), len(self.step_names))
        
        for i in range(executed_steps):
            time_taken = self.step_times[i]
            name = self.step_names[i]
            summary.append(
                f"  Step {i+1}: {name.ljust(25)} "
                f"{time_taken:.2f}s ({time_taken/total_time:.1%})"
            )

        if self._execution_stopped and self.max_steps is not None:
            summary.append(f"\n ⏹️ Execution stopped after step {self.max_steps}")
            
        summary.append(f"\n ⏱️ Total processing time: {total_time:.2f}s")
        summary.append("="*50)

        full_summary = "\n".join(summary)
        # print(full_summary)
        if self.logger:
            self.logger.info(full_summary)
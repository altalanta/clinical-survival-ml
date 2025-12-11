"""
Pipeline checkpoint and resume functionality.

This module enables:
- Saving pipeline state after each step
- Resuming from the last successful step after failures
- Checkpoint management and cleanup

Usage:
    from clinical_survival.checkpoint import CheckpointManager
    
    # Create manager
    manager = CheckpointManager("results/checkpoints")
    
    # Save checkpoint
    manager.save_checkpoint("data_loader", context)
    
    # Resume from checkpoint
    context = manager.load_latest_checkpoint()
    resume_from = manager.get_last_successful_step()
"""

from __future__ import annotations

import json
import pickle
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from clinical_survival.logging_config import get_logger
from clinical_survival.utils import ensure_dir

# Module logger
logger = get_logger(__name__)


@dataclass
class CheckpointInfo:
    """Information about a checkpoint."""
    
    step_name: str
    step_index: int
    created_at: str
    run_id: str
    correlation_id: Optional[str] = None
    context_keys: List[str] = field(default_factory=list)
    file_path: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return self.__dict__.copy()
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CheckpointInfo":
        """Create from dictionary."""
        return cls(**data)


@dataclass
class CheckpointState:
    """State of the checkpoint system for a run."""
    
    run_id: str
    correlation_id: Optional[str] = None
    pipeline_steps: List[str] = field(default_factory=list)
    completed_steps: List[str] = field(default_factory=list)
    checkpoints: List[CheckpointInfo] = field(default_factory=list)
    status: str = "running"  # running, completed, failed
    started_at: Optional[str] = None
    updated_at: Optional[str] = None
    error_message: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "run_id": self.run_id,
            "correlation_id": self.correlation_id,
            "pipeline_steps": self.pipeline_steps,
            "completed_steps": self.completed_steps,
            "checkpoints": [c.to_dict() for c in self.checkpoints],
            "status": self.status,
            "started_at": self.started_at,
            "updated_at": self.updated_at,
            "error_message": self.error_message,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CheckpointState":
        """Create from dictionary."""
        checkpoints = [CheckpointInfo.from_dict(c) for c in data.pop("checkpoints", [])]
        state = cls(**data)
        state.checkpoints = checkpoints
        return state


class CheckpointManager:
    """
    Manages pipeline checkpoints for resume capability.
    
    Usage:
        manager = CheckpointManager("results/checkpoints", run_id="abc123")
        
        # Start a new run
        manager.start_run(pipeline_steps=["step1", "step2", "step3"])
        
        # After each step
        manager.save_checkpoint("step1", context)
        manager.mark_step_completed("step1")
        
        # If resuming after failure
        context = manager.load_checkpoint("step1")
        resume_from = manager.get_resume_step()
    """
    
    # Keys in context that should not be pickled
    NON_SERIALIZABLE_KEYS: Set[str] = {
        "tracker",  # MLflow tracker
        "memory",   # Joblib memory
    }
    
    def __init__(
        self,
        checkpoint_dir: Path,
        run_id: Optional[str] = None,
        max_checkpoints: int = 10,
    ):
        """
        Initialize the checkpoint manager.
        
        Args:
            checkpoint_dir: Directory to store checkpoints
            run_id: Optional run ID (generated if not provided)
            max_checkpoints: Maximum checkpoints to keep per run
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.run_id = run_id or self._generate_run_id()
        self.max_checkpoints = max_checkpoints
        
        self._state: Optional[CheckpointState] = None
        self._run_dir: Optional[Path] = None
    
    def _generate_run_id(self) -> str:
        """Generate a unique run ID."""
        import uuid
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        return f"{timestamp}_{uuid.uuid4().hex[:8]}"
    
    @property
    def run_dir(self) -> Path:
        """Get the directory for this run's checkpoints."""
        if self._run_dir is None:
            self._run_dir = self.checkpoint_dir / self.run_id
            ensure_dir(self._run_dir)
        return self._run_dir
    
    @property
    def state_file(self) -> Path:
        """Path to the state file."""
        return self.run_dir / "state.json"
    
    def start_run(
        self,
        pipeline_steps: List[str],
        correlation_id: Optional[str] = None,
    ) -> None:
        """
        Start a new run with the given pipeline steps.
        
        Args:
            pipeline_steps: List of pipeline step names in order
            correlation_id: Optional correlation ID
        """
        self._state = CheckpointState(
            run_id=self.run_id,
            correlation_id=correlation_id,
            pipeline_steps=pipeline_steps,
            started_at=datetime.utcnow().isoformat(),
            updated_at=datetime.utcnow().isoformat(),
        )
        
        self._save_state()
        
        logger.info(
            f"Started checkpoint run: {self.run_id}",
            extra={"run_id": self.run_id, "n_steps": len(pipeline_steps)},
        )
    
    def _save_state(self) -> None:
        """Save current state to disk."""
        if self._state is None:
            return
        
        self._state.updated_at = datetime.utcnow().isoformat()
        
        with open(self.state_file, "w") as f:
            json.dump(self._state.to_dict(), f, indent=2)
    
    def _load_state(self) -> Optional[CheckpointState]:
        """Load state from disk."""
        if not self.state_file.exists():
            return None
        
        with open(self.state_file) as f:
            data = json.load(f)
        
        return CheckpointState.from_dict(data)
    
    def save_checkpoint(
        self,
        step_name: str,
        context: Dict[str, Any],
    ) -> Path:
        """
        Save a checkpoint after a step completes.
        
        Args:
            step_name: Name of the completed step
            context: Pipeline context dictionary
            
        Returns:
            Path to the checkpoint file
        """
        if self._state is None:
            self._state = self._load_state()
            if self._state is None:
                raise RuntimeError("No run started. Call start_run() first.")
        
        # Filter non-serializable objects
        serializable_context = {
            k: v for k, v in context.items()
            if k not in self.NON_SERIALIZABLE_KEYS
        }
        
        # Create checkpoint file
        step_index = len(self._state.completed_steps)
        checkpoint_file = self.run_dir / f"checkpoint_{step_index:02d}_{step_name}.pkl"
        
        with open(checkpoint_file, "wb") as f:
            pickle.dump(serializable_context, f)
        
        # Create checkpoint info
        info = CheckpointInfo(
            step_name=step_name,
            step_index=step_index,
            created_at=datetime.utcnow().isoformat(),
            run_id=self.run_id,
            correlation_id=self._state.correlation_id,
            context_keys=list(serializable_context.keys()),
            file_path=str(checkpoint_file),
        )
        
        self._state.checkpoints.append(info)
        self._save_state()
        
        logger.debug(
            f"Saved checkpoint: {step_name}",
            extra={"step": step_name, "path": str(checkpoint_file)},
        )
        
        # Cleanup old checkpoints if needed
        self._cleanup_old_checkpoints()
        
        return checkpoint_file
    
    def mark_step_completed(self, step_name: str) -> None:
        """Mark a step as completed."""
        if self._state is None:
            self._state = self._load_state()
            if self._state is None:
                raise RuntimeError("No run started")
        
        if step_name not in self._state.completed_steps:
            self._state.completed_steps.append(step_name)
            self._save_state()
        
        logger.debug(f"Marked step completed: {step_name}")
    
    def mark_run_completed(self) -> None:
        """Mark the run as completed."""
        if self._state is None:
            self._state = self._load_state()
        
        if self._state:
            self._state.status = "completed"
            self._save_state()
            logger.info(f"Run completed: {self.run_id}")
    
    def mark_run_failed(self, error_message: str) -> None:
        """Mark the run as failed."""
        if self._state is None:
            self._state = self._load_state()
        
        if self._state:
            self._state.status = "failed"
            self._state.error_message = error_message
            self._save_state()
            logger.info(f"Run failed: {self.run_id} - {error_message}")
    
    def load_checkpoint(self, step_name: str) -> Optional[Dict[str, Any]]:
        """
        Load a specific checkpoint.
        
        Args:
            step_name: Name of the step to load
            
        Returns:
            Context dictionary or None if not found
        """
        if self._state is None:
            self._state = self._load_state()
        
        if self._state is None:
            return None
        
        # Find the checkpoint
        for checkpoint in reversed(self._state.checkpoints):
            if checkpoint.step_name == step_name and checkpoint.file_path:
                path = Path(checkpoint.file_path)
                if path.exists():
                    with open(path, "rb") as f:
                        context = pickle.load(f)
                    logger.info(f"Loaded checkpoint: {step_name}")
                    return context
        
        return None
    
    def load_latest_checkpoint(self) -> Optional[Dict[str, Any]]:
        """
        Load the most recent checkpoint.
        
        Returns:
            Context dictionary or None if no checkpoints exist
        """
        if self._state is None:
            self._state = self._load_state()
        
        if self._state is None or not self._state.checkpoints:
            return None
        
        latest = self._state.checkpoints[-1]
        return self.load_checkpoint(latest.step_name)
    
    def get_resume_step(self) -> Optional[str]:
        """
        Get the step to resume from.
        
        Returns:
            Name of the next step to run, or None if complete
        """
        if self._state is None:
            self._state = self._load_state()
        
        if self._state is None:
            return None
        
        if not self._state.completed_steps:
            return self._state.pipeline_steps[0] if self._state.pipeline_steps else None
        
        last_completed = self._state.completed_steps[-1]
        
        try:
            idx = self._state.pipeline_steps.index(last_completed)
            if idx + 1 < len(self._state.pipeline_steps):
                return self._state.pipeline_steps[idx + 1]
        except ValueError:
            pass
        
        return None
    
    def get_completed_steps(self) -> List[str]:
        """Get list of completed steps."""
        if self._state is None:
            self._state = self._load_state()
        
        return self._state.completed_steps if self._state else []
    
    def can_resume(self) -> bool:
        """Check if the run can be resumed."""
        if self._state is None:
            self._state = self._load_state()
        
        if self._state is None:
            return False
        
        return (
            self._state.status == "failed"
            and len(self._state.completed_steps) > 0
            and len(self._state.checkpoints) > 0
        )
    
    def _cleanup_old_checkpoints(self) -> None:
        """Remove old checkpoints if over the limit."""
        if self._state is None or len(self._state.checkpoints) <= self.max_checkpoints:
            return
        
        # Keep only the most recent checkpoints
        to_remove = self._state.checkpoints[:-self.max_checkpoints]
        
        for checkpoint in to_remove:
            if checkpoint.file_path:
                path = Path(checkpoint.file_path)
                if path.exists():
                    path.unlink()
        
        self._state.checkpoints = self._state.checkpoints[-self.max_checkpoints:]
        self._save_state()
    
    def cleanup(self) -> None:
        """Remove all checkpoints for this run."""
        if self.run_dir.exists():
            shutil.rmtree(self.run_dir)
            logger.info(f"Cleaned up checkpoints for run: {self.run_id}")
    
    @classmethod
    def list_runs(cls, checkpoint_dir: Path) -> List[CheckpointState]:
        """
        List all runs with checkpoints.
        
        Args:
            checkpoint_dir: Directory containing checkpoints
            
        Returns:
            List of CheckpointState objects
        """
        runs = []
        checkpoint_dir = Path(checkpoint_dir)
        
        if not checkpoint_dir.exists():
            return runs
        
        for run_dir in checkpoint_dir.iterdir():
            if run_dir.is_dir():
                state_file = run_dir / "state.json"
                if state_file.exists():
                    with open(state_file) as f:
                        data = json.load(f)
                    runs.append(CheckpointState.from_dict(data))
        
        # Sort by most recent first
        runs.sort(key=lambda x: x.updated_at or "", reverse=True)
        
        return runs
    
    @classmethod
    def get_resumable_run(cls, checkpoint_dir: Path) -> Optional["CheckpointManager"]:
        """
        Get the most recent resumable run.
        
        Args:
            checkpoint_dir: Directory containing checkpoints
            
        Returns:
            CheckpointManager for the resumable run, or None
        """
        runs = cls.list_runs(checkpoint_dir)
        
        for state in runs:
            if state.status == "failed" and state.checkpoints:
                manager = cls(checkpoint_dir, run_id=state.run_id)
                manager._state = state
                return manager
        
        return None


def create_checkpoint_manager(
    output_dir: Path,
    run_id: Optional[str] = None,
) -> CheckpointManager:
    """
    Create a checkpoint manager for the given output directory.
    
    Args:
        output_dir: Pipeline output directory
        run_id: Optional run ID
        
    Returns:
        CheckpointManager instance
    """
    checkpoint_dir = Path(output_dir) / "checkpoints"
    return CheckpointManager(checkpoint_dir, run_id=run_id)






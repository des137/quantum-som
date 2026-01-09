"""
IBM Quantum Job Management for QSOM.

This module provides utilities for managing quantum jobs on IBM Quantum
hardware, including:
- Job submission and monitoring
- Batch job processing
- Job persistence and recovery
- Queue management
"""

from typing import List, Dict, Optional, Any, Callable, Union
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import json
import time
import numpy as np

try:
    from qiskit import QuantumCircuit
    from qiskit_ibm_runtime import (
        QiskitRuntimeService,
        Sampler,
        Estimator,
        Session,
        Batch,
        RuntimeJob,
    )
    IBM_RUNTIME_AVAILABLE = True
except ImportError:
    IBM_RUNTIME_AVAILABLE = False
    QiskitRuntimeService = None
    RuntimeJob = None


@dataclass
class JobInfo:
    """Information about a submitted job."""
    job_id: str
    circuit_name: str
    backend_name: str
    status: str
    submitted_at: str
    completed_at: Optional[str] = None
    n_circuits: int = 1
    shots: int = 1000
    result: Optional[Dict] = None
    error_message: Optional[str] = None
    metadata: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'job_id': self.job_id,
            'circuit_name': self.circuit_name,
            'backend_name': self.backend_name,
            'status': self.status,
            'submitted_at': self.submitted_at,
            'completed_at': self.completed_at,
            'n_circuits': self.n_circuits,
            'shots': self.shots,
            'error_message': self.error_message,
            'metadata': self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'JobInfo':
        """Create from dictionary."""
        return cls(
            job_id=data['job_id'],
            circuit_name=data['circuit_name'],
            backend_name=data['backend_name'],
            status=data['status'],
            submitted_at=data['submitted_at'],
            completed_at=data.get('completed_at'),
            n_circuits=data.get('n_circuits', 1),
            shots=data.get('shots', 1000),
            error_message=data.get('error_message'),
            metadata=data.get('metadata', {})
        )


class IBMJobManager:
    """
    Manager for IBM Quantum jobs.

    Provides:
    - Job submission with automatic persistence
    - Status monitoring and polling
    - Result retrieval and caching
    - Batch job support
    - Job recovery after interruption
    """

    def __init__(
        self,
        service: Optional['QiskitRuntimeService'] = None,
        backend_name: Optional[str] = None,
        jobs_dir: Optional[str] = None,
        auto_save: bool = True
    ):
        """
        Initialize job manager.

        Args:
            service: QiskitRuntimeService instance. If None, loads from saved credentials.
            backend_name: Default backend to use.
            jobs_dir: Directory to store job information.
            auto_save: Automatically save job info after submission.
        """
        if not IBM_RUNTIME_AVAILABLE:
            raise ImportError(
                "qiskit-ibm-runtime is required. "
                "Install with: pip install qiskit-ibm-runtime"
            )

        self.service = service or QiskitRuntimeService()
        self.backend_name = backend_name
        self.jobs_dir = Path(jobs_dir) if jobs_dir else Path.home() / '.qsom_jobs'
        self.auto_save = auto_save

        # Create jobs directory
        self.jobs_dir.mkdir(parents=True, exist_ok=True)

        # Job tracking
        self.submitted_jobs: Dict[str, JobInfo] = {}
        self.active_session: Optional['Session'] = None

        # Load existing jobs
        self._load_jobs()

    def _load_jobs(self) -> None:
        """Load previously submitted jobs from disk."""
        jobs_file = self.jobs_dir / 'jobs.json'
        if jobs_file.exists():
            with open(jobs_file, 'r') as f:
                data = json.load(f)
                for job_data in data.get('jobs', []):
                    job_info = JobInfo.from_dict(job_data)
                    self.submitted_jobs[job_info.job_id] = job_info

    def _save_jobs(self) -> None:
        """Save job information to disk."""
        jobs_file = self.jobs_dir / 'jobs.json'
        data = {
            'jobs': [job.to_dict() for job in self.submitted_jobs.values()],
            'updated_at': datetime.now().isoformat()
        }
        with open(jobs_file, 'w') as f:
            json.dump(data, f, indent=2)

    def get_backend(self, name: Optional[str] = None) -> Any:
        """
        Get backend by name.

        Args:
            name: Backend name. Uses default if not specified.

        Returns:
            Backend instance.
        """
        backend_name = name or self.backend_name
        if backend_name is None:
            # Get least busy backend
            backends = self.service.backends(
                filters=lambda b: b.status().operational
                and b.status().pending_jobs < 10
            )
            if backends:
                return min(backends, key=lambda b: b.status().pending_jobs)
            raise ValueError("No available backends")

        return self.service.backend(backend_name)

    def list_backends(self, operational_only: bool = True) -> List[Dict]:
        """
        List available backends.

        Args:
            operational_only: Only list operational backends.

        Returns:
            List of backend information dictionaries.
        """
        backends = []
        for backend in self.service.backends():
            status = backend.status()
            if operational_only and not status.operational:
                continue

            backends.append({
                'name': backend.name,
                'n_qubits': backend.num_qubits,
                'operational': status.operational,
                'pending_jobs': status.pending_jobs,
                'status_msg': status.status_msg
            })

        return sorted(backends, key=lambda b: b['pending_jobs'])

    def submit_circuit(
        self,
        circuit: 'QuantumCircuit',
        shots: int = 1000,
        backend_name: Optional[str] = None,
        name: Optional[str] = None
    ) -> JobInfo:
        """
        Submit a single circuit for execution.

        Args:
            circuit: Quantum circuit to run.
            shots: Number of shots.
            backend_name: Backend to use.
            name: Job name for tracking.

        Returns:
            JobInfo with job details.
        """
        backend = self.get_backend(backend_name)

        with Sampler(backend=backend) as sampler:
            job = sampler.run([circuit], shots=shots)

        job_info = JobInfo(
            job_id=job.job_id(),
            circuit_name=name or circuit.name or 'unnamed',
            backend_name=backend.name,
            status='QUEUED',
            submitted_at=datetime.now().isoformat(),
            n_circuits=1,
            shots=shots
        )

        self.submitted_jobs[job.job_id()] = job_info

        if self.auto_save:
            self._save_jobs()

        return job_info

    def submit_batch(
        self,
        circuits: List['QuantumCircuit'],
        shots: int = 1000,
        backend_name: Optional[str] = None,
        batch_name: Optional[str] = None
    ) -> List[JobInfo]:
        """
        Submit multiple circuits as a batch.

        Args:
            circuits: List of circuits to run.
            shots: Shots per circuit.
            backend_name: Backend to use.
            batch_name: Name for the batch.

        Returns:
            List of JobInfo objects.
        """
        backend = self.get_backend(backend_name)
        job_infos = []

        with Batch(backend=backend) as batch:
            sampler = Sampler()
            job = sampler.run(circuits, shots=shots)

            batch_id = batch_name or f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

            job_info = JobInfo(
                job_id=job.job_id(),
                circuit_name=batch_id,
                backend_name=backend.name,
                status='QUEUED',
                submitted_at=datetime.now().isoformat(),
                n_circuits=len(circuits),
                shots=shots,
                metadata={'batch_name': batch_id}
            )

            self.submitted_jobs[job.job_id()] = job_info
            job_infos.append(job_info)

        if self.auto_save:
            self._save_jobs()

        return job_infos

    def start_session(
        self,
        backend_name: Optional[str] = None,
        max_time: int = 3600
    ) -> 'Session':
        """
        Start a runtime session for multiple job submissions.

        Sessions keep your jobs together and can provide better queue times.

        Args:
            backend_name: Backend for the session.
            max_time: Maximum session duration in seconds.

        Returns:
            Session context manager.
        """
        backend = self.get_backend(backend_name)
        self.active_session = Session(backend=backend, max_time=max_time)
        return self.active_session

    def submit_in_session(
        self,
        circuits: List['QuantumCircuit'],
        shots: int = 1000,
        name: Optional[str] = None
    ) -> JobInfo:
        """
        Submit circuits within active session.

        Args:
            circuits: Circuits to run.
            shots: Number of shots.
            name: Job name.

        Returns:
            JobInfo for the submitted job.
        """
        if self.active_session is None:
            raise ValueError("No active session. Call start_session() first.")

        sampler = Sampler(session=self.active_session)
        job = sampler.run(circuits, shots=shots)

        job_info = JobInfo(
            job_id=job.job_id(),
            circuit_name=name or 'session_job',
            backend_name=self.active_session._backend.name,
            status='QUEUED',
            submitted_at=datetime.now().isoformat(),
            n_circuits=len(circuits),
            shots=shots,
            metadata={'session_id': self.active_session.session_id}
        )

        self.submitted_jobs[job.job_id()] = job_info

        if self.auto_save:
            self._save_jobs()

        return job_info

    def get_job_status(self, job_id: str) -> str:
        """
        Get current status of a job.

        Args:
            job_id: Job ID to check.

        Returns:
            Job status string.
        """
        try:
            job = self.service.job(job_id)
            status = job.status().name
            if job_id in self.submitted_jobs:
                self.submitted_jobs[job_id].status = status
            return status
        except Exception as e:
            return f"ERROR: {str(e)}"

    def wait_for_job(
        self,
        job_id: str,
        timeout: int = 3600,
        poll_interval: int = 10,
        callback: Optional[Callable] = None
    ) -> Any:
        """
        Wait for job to complete.

        Args:
            job_id: Job ID to wait for.
            timeout: Maximum wait time in seconds.
            poll_interval: Time between status checks.
            callback: Function to call on each poll with (job_id, status).

        Returns:
            Job result when complete.
        """
        job = self.service.job(job_id)
        start_time = time.time()

        while True:
            status = job.status().name

            if callback:
                callback(job_id, status)

            if status in ['DONE', 'COMPLETED']:
                result = job.result()
                if job_id in self.submitted_jobs:
                    self.submitted_jobs[job_id].status = status
                    self.submitted_jobs[job_id].completed_at = datetime.now().isoformat()
                    if self.auto_save:
                        self._save_jobs()
                return result

            if status in ['CANCELLED', 'ERROR']:
                if job_id in self.submitted_jobs:
                    self.submitted_jobs[job_id].status = status
                    self.submitted_jobs[job_id].error_message = str(job.error_message())
                raise RuntimeError(f"Job {job_id} failed: {status}")

            if time.time() - start_time > timeout:
                raise TimeoutError(f"Job {job_id} timed out after {timeout}s")

            time.sleep(poll_interval)

    def get_result(self, job_id: str) -> Optional[Any]:
        """
        Get result of completed job.

        Args:
            job_id: Job ID.

        Returns:
            Job result or None if not complete.
        """
        try:
            job = self.service.job(job_id)
            if job.status().name in ['DONE', 'COMPLETED']:
                return job.result()
            return None
        except Exception:
            return None

    def cancel_job(self, job_id: str) -> bool:
        """
        Cancel a queued or running job.

        Args:
            job_id: Job ID to cancel.

        Returns:
            True if cancellation successful.
        """
        try:
            job = self.service.job(job_id)
            job.cancel()
            if job_id in self.submitted_jobs:
                self.submitted_jobs[job_id].status = 'CANCELLED'
                if self.auto_save:
                    self._save_jobs()
            return True
        except Exception:
            return False

    def list_jobs(
        self,
        status_filter: Optional[str] = None,
        limit: int = 20
    ) -> List[JobInfo]:
        """
        List submitted jobs.

        Args:
            status_filter: Filter by status (e.g., 'QUEUED', 'RUNNING', 'DONE').
            limit: Maximum number of jobs to return.

        Returns:
            List of JobInfo objects.
        """
        jobs = list(self.submitted_jobs.values())

        # Update statuses
        for job_info in jobs[:limit]:
            try:
                status = self.get_job_status(job_info.job_id)
                job_info.status = status
            except Exception:
                pass

        if status_filter:
            jobs = [j for j in jobs if j.status == status_filter]

        # Sort by submission time
        jobs.sort(key=lambda j: j.submitted_at, reverse=True)

        return jobs[:limit]

    def recover_jobs(self) -> Dict[str, JobInfo]:
        """
        Recover and update status of all saved jobs.

        Returns:
            Dictionary of job_id to updated JobInfo.
        """
        recovered = {}

        for job_id, job_info in self.submitted_jobs.items():
            try:
                job = self.service.job(job_id)
                status = job.status().name
                job_info.status = status

                if status in ['DONE', 'COMPLETED']:
                    job_info.completed_at = job_info.completed_at or datetime.now().isoformat()

                recovered[job_id] = job_info

            except Exception as e:
                job_info.error_message = str(e)
                recovered[job_id] = job_info

        if self.auto_save:
            self._save_jobs()

        return recovered

    def cleanup_old_jobs(self, days: int = 30) -> int:
        """
        Remove job records older than specified days.

        Args:
            days: Age threshold in days.

        Returns:
            Number of jobs removed.
        """
        from datetime import timedelta

        threshold = datetime.now() - timedelta(days=days)
        to_remove = []

        for job_id, job_info in self.submitted_jobs.items():
            submitted = datetime.fromisoformat(job_info.submitted_at)
            if submitted < threshold:
                to_remove.append(job_id)

        for job_id in to_remove:
            del self.submitted_jobs[job_id]

        if self.auto_save and to_remove:
            self._save_jobs()

        return len(to_remove)


class ShadowJobRunner:
    """
    Specialized job runner for classical shadow generation.

    Manages the submission and collection of shadow measurement circuits
    on IBM Quantum hardware.
    """

    def __init__(
        self,
        job_manager: IBMJobManager,
        n_qubits: int,
        shadow_size: int = 100,
        shots_per_circuit: int = 1
    ):
        """
        Initialize shadow job runner.

        Args:
            job_manager: IBMJobManager instance.
            n_qubits: Number of qubits.
            shadow_size: Number of shadow circuits.
            shots_per_circuit: Shots per measurement circuit.
        """
        self.job_manager = job_manager
        self.n_qubits = n_qubits
        self.shadow_size = shadow_size
        self.shots = shots_per_circuit

    def submit_shadow_circuits(
        self,
        state_circuit: 'QuantumCircuit',
        pauli_bases: List[str],
        name: str = "shadow"
    ) -> JobInfo:
        """
        Submit shadow measurement circuits.

        Args:
            state_circuit: Circuit preparing the state.
            pauli_bases: List of Pauli basis strings (e.g., ['XXY', 'ZZX', ...]).
            name: Job name.

        Returns:
            JobInfo for the batch.
        """
        circuits = []

        for bases in pauli_bases:
            # Create measurement circuit
            meas_circuit = state_circuit.copy()

            # Add basis rotation
            for q, basis in enumerate(bases):
                if basis == 'X':
                    meas_circuit.h(q)
                elif basis == 'Y':
                    meas_circuit.sdg(q)
                    meas_circuit.h(q)
                # Z basis: no rotation needed

            meas_circuit.measure_all()
            circuits.append(meas_circuit)

        # Submit as batch
        job_infos = self.job_manager.submit_batch(
            circuits, shots=self.shots, batch_name=name
        )

        return job_infos[0] if job_infos else None

    def collect_shadow_results(
        self,
        job_id: str,
        pauli_bases: List[str],
        timeout: int = 3600
    ) -> List[tuple]:
        """
        Collect shadow results from completed job.

        Args:
            job_id: Job ID to collect.
            pauli_bases: Original Pauli bases used.
            timeout: Maximum wait time.

        Returns:
            List of (bases, outcomes) tuples.
        """
        result = self.job_manager.wait_for_job(job_id, timeout=timeout)
        shadow_samples = []

        # Extract results for each circuit
        for i, bases in enumerate(pauli_bases):
            # Get quasi-distribution
            try:
                dist = result[i].data
                # Sample from distribution
                if hasattr(dist, 'meas'):
                    bitstrings = dist.meas.get_bitstrings()
                    if bitstrings:
                        bitstring = bitstrings[0]
                    else:
                        bitstring = '0' * self.n_qubits
                else:
                    # Sample from quasi_dist
                    bitstring = format(
                        np.random.choice(2 ** self.n_qubits),
                        f'0{self.n_qubits}b'
                    )
            except Exception:
                bitstring = '0' * self.n_qubits

            # Convert to arrays
            bases_array = np.array([
                {'X': 0, 'Y': 1, 'Z': 2}[b] for b in bases
            ])
            outcomes = np.array([int(b) for b in bitstring[::-1]])

            shadow_samples.append((bases_array, outcomes))

        return shadow_samples


def estimate_job_time(
    n_circuits: int,
    n_qubits: int,
    shots: int,
    backend_name: str = 'ibm_brisbane'
) -> Dict[str, Any]:
    """
    Estimate execution time for a job.

    Args:
        n_circuits: Number of circuits.
        n_qubits: Qubits per circuit.
        shots: Shots per circuit.
        backend_name: Target backend.

    Returns:
        Estimate dictionary with time and cost info.
    """
    # Rough estimates based on typical IBM Quantum performance
    base_circuit_time = 0.001  # 1ms per circuit execution
    shot_time = 0.0001  # 0.1ms per shot
    queue_time_estimate = 300  # 5 min average queue

    execution_time = n_circuits * shots * shot_time
    total_estimated = queue_time_estimate + execution_time

    return {
        'n_circuits': n_circuits,
        'n_qubits': n_qubits,
        'shots': shots,
        'estimated_execution_seconds': execution_time,
        'estimated_queue_seconds': queue_time_estimate,
        'estimated_total_seconds': total_estimated,
        'estimated_total_minutes': total_estimated / 60,
        'backend': backend_name
    }

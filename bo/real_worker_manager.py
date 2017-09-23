"""
  Worker manager for Resnet.
  -- kandasamy@cs.cmu.edu
"""
# pylint: disable=invalid-name

import os
import shutil
import time
from sets import Set
from multiprocessing import Process

from bo.worker_manager import WorkerManager

class RealWorkerManager(WorkerManager):
  """ A worker manager for Real time parallel workers. """
  # pylint: disable=attribute-defined-outside-init

  def __init__(self, func_caller, worker_ids, poll_time=1):
    """ Constructor. """
    super(RealWorkerManager, self).__init__(func_caller, worker_ids, poll_time)
    self._rwm_set_up()
    self._child_reset()

  def _rwm_set_up(self):
    """ Sets things up for the child. """
    # Create the result directories. """
    self.result_dir_names = {wid:'exp/result_%s'%(str(wid)) for wid in
                                                      self.worker_ids}
    # Create the working directories
    self.working_dir_names = {wid:'exp/working_%s/tmp'%(str(wid)) for wid in
                                                            self.worker_ids}
    # Create the last receive times
    self.last_receive_times = {wid:0.0 for wid in self.worker_ids}
    # Create file names
    self._result_file_name = 'result.txt'
    self._num_file_read_attempts = 10
    self._file_read_poll_time = 0.5 # wait for 0.5 seconds

  @classmethod
  def _delete_dirs(cls, list_of_dir_names):
    """ Deletes a list of directories. """
    for dir_name in list_of_dir_names:
      if os.path.exists(dir_name):
        shutil.rmtree(dir_name)

  @classmethod
  def _delete_and_create_dirs(cls, list_of_dir_names):
    """ Deletes a list of directories and creates new ones. """
    for dir_name in list_of_dir_names:
      if os.path.exists(dir_name):
        shutil.rmtree(dir_name)
      os.makedirs(dir_name)

  def _child_reset(self):
    """ Resets child. """
    # Delete/create the result and working directories.
    if not hasattr(self, 'result_dir_names'): # Just for the super constructor.
      return
    self._delete_and_create_dirs(self.result_dir_names.values())
    self._delete_dirs(self.working_dir_names.values())
    self.free_workers = Set(self.worker_ids)
    self.qinfos_in_progress = {wid:None for wid in self.worker_ids}
    self.worker_processes = {wid:None for wid in self.worker_ids}

  def _get_result_file_name_for_worker(self, worker_id):
    """ Computes the result file name for the worker. """
    return os.path.join(self.result_dir_names[worker_id], self._result_file_name)

  def _read_result_from_file(self, result_file_name):
    """ Reads the result from the file name. """
    #pylint: disable=bare-except
    num_attempts = 0
    result = 0.5
    while num_attempts < self._num_file_read_attempts:
      try:
        file_reader = open(result_file_name, 'r')
        read_in = float(file_reader.read().strip())
        file_reader.close()
        result = read_in
        break
      except:
        print 'Encountered error when reading %s. Trying again.'%(result_file_name)
        time.sleep(self._file_read_poll_time)
        file_reader.close()
    return result

  def _read_result_from_worker_and_update(self, worker_id):
    """ Reads the result from the worker. """
    # Read the file
    result_file_name = self._get_result_file_name_for_worker(worker_id)
    val = self._read_result_from_file(result_file_name)
    # Now update the relevant qinfo and put it to latest_results
    qinfo = self.qinfos_in_progress[worker_id]
    qinfo.val = val
    if not hasattr(qinfo, 'true_val'):
      qinfo.true_val = val
    qinfo.receive_time = self.optimiser.get_curr_spent_capital()
    qinfo.eval_time = qinfo.receive_time - qinfo.send_time
    self.latest_results.append(qinfo)
    # Update receive time
    self.last_receive_times[worker_id] = qinfo.receive_time
    # Delete the file.
    os.remove(result_file_name)
    # Delete content in a working directory.
    shutil.rmtree(self.working_dir_names[worker_id])
    # Add the worker to the list of free workers and clear qinfos in progress.
    self.worker_processes[worker_id].terminate()
    self.worker_processes[worker_id] = None
    self.qinfos_in_progress[worker_id] = None
    self.free_workers.add(worker_id)

  def _worker_is_free(self, worker_id):
    """ Checks if worker with worker_id is free. """
    if worker_id in self.free_workers:
      return True
    worker_result_file_name = self._get_result_file_name_for_worker(worker_id)
    if os.path.exists(worker_result_file_name):
      self._read_result_from_worker_and_update(worker_id)
    else:
      return False

  def _get_last_receive_time(self):
    """ Returns the last time we received a job. """
    all_receive_times = self.last_receive_times.values()
    return max(all_receive_times)

  def a_worker_is_free(self):
    """ Returns true if a worker is free. """
    for wid in self.worker_ids:
      if self._worker_is_free(wid):
        return self._get_last_receive_time()
    return None

  def all_workers_are_free(self):
    """ Returns true if all workers are free. """
    all_are_free = True
    for wid in self.worker_ids:
      all_are_free = self._worker_is_free(wid) and all_are_free
    if all_are_free:
      return self._get_last_receive_time()
    else:
      return None

  def _dispatch_evaluation(self, func_caller, point, qinfo, worker_id, **kwargs):
    """ Dispatches evaluation to worker_id. """
    #pylint: disable=star-args
    if self.qinfos_in_progress[worker_id] is not None:
      err_msg = 'qinfos_in_progress: %s,\nfree_workers: %s.'%(
                   str(self.qinfos_in_progress), str(self.free_workers))
      print err_msg
      raise ValueError('Check if worker is free before sending evaluation.')
    # First add all the data to qinfo
    qinfo.worker_id = worker_id
    qinfo.working_dir = self.working_dir_names[worker_id]
    qinfo.result_file = self._get_result_file_name_for_worker(worker_id)
    qinfo.point = point
    # Create the working directory
    os.makedirs(qinfo.working_dir)
    # Dispatch the evaluation in a new process
    target_func = lambda: func_caller.eval_single(point, qinfo, **kwargs)
    self.worker_processes[worker_id] = Process(target=target_func)
    self.worker_processes[worker_id].start()
    # Add the qinfo to the in progress bar and remove from free_workers
    self.qinfos_in_progress[worker_id] = qinfo
    self.free_workers.discard(worker_id)

  def dispatch_single_evaluation(self, func_caller, point, qinfo, **kwargs):
    """ Dispatches a single evaluation to a free worker. """
    worker_id = self.free_workers.pop()
    self._dispatch_evaluation(func_caller, point, qinfo, worker_id, **kwargs)

  def dispatch_batch_of_evaluations(self, func_caller, points, qinfos, **kwargs):
    """ Dispatches a batch of evaluations. """
    assert len(points) == self.num_workers
    for idx in range(self.num_workers):
      self._dispatch_evaluation(func_caller, points[idx], qinfos[idx],
                                self.worker_ids[idx], **kwargs)

  def close_all_jobs(self):
    """ Closes all jobs. """
    pass


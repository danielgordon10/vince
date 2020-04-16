import torch.multiprocessing as multiprocessing

from end_tasks import end_task_eval

cxt = multiprocessing.get_context()
print(cxt.get_start_method())
end_task_eval.main()

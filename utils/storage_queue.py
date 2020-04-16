import torch


class StorageQueue(object):
    def __init__(self, maxsize, feat_size, device=None, dtype=torch.float32):
        self.maxsize = maxsize
        self.feat_size = feat_size
        self.device = device
        self.dtype = dtype
        self.vector_queue = torch.nn.functional.normalize(
            torch.randn((maxsize, feat_size), device=device, requires_grad=False, dtype=dtype), dim=-1
        )
        self.image_queue = [None for _ in range(maxsize)]
        self.data_source_queue = [None for _ in range(maxsize)]
        self.current_tail = 0
        self.full = False

    def __len__(self):
        return len(self.image_queue)

    def clear(self):
        self.vector_queue = torch.nn.functional.normalize(
            torch.randn((self.maxsize, self.feat_size), device=self.device, requires_grad=False, dtype=self.dtype),
            dim=-1,
        )
        self.image_queue = [None for _ in range(self.maxsize)]
        self.data_source_queue = [None for _ in range(self.maxsize)]
        self.current_tail = 0
        self.full = False

    def enqueue(self, items, item_images, data_source):
        assert len(items) == len(item_images)
        with torch.no_grad():
            num_items = items.shape[0]
            if self.current_tail + num_items > self.maxsize:
                num_start = self.maxsize - self.current_tail
                if num_start > 0:
                    self.vector_queue[self.current_tail:].copy_(items[:num_start])
                    self.image_queue[self.current_tail:] = item_images[:num_start]
                    self.data_source_queue[self.current_tail:] = [data_source] * num_start
                self.current_tail = 0
                self.full = True
                self.enqueue(items[num_start:], item_images[num_start:], data_source)
            else:
                # Shift the buffer
                self.vector_queue[self.current_tail: self.current_tail + num_items].copy_(items)
                self.image_queue[self.current_tail: self.current_tail + num_items] = item_images
                self.data_source_queue[self.current_tail: self.current_tail + num_items] = [data_source] * num_items
                self.current_tail += num_items

    def dequeue(self):
        return {
            "queue_vectors": self.vector_queue.detach(),
            "queue_images": self.image_queue,
            "queue_data_sources": self.data_source_queue,
        }

import multiprocessing as mp
import random

import tqdm
from dg_util.python_utils import misc_util
from dg_util.python_utils import youtube_utils

DEBUG = False
num_threads = 800

queue = mp.Queue()


def search(search_tuple: str, num_results=1000):
    synset_id, search_text = search_tuple
    video_urls = []
    count = 0
    while len(video_urls) == 0 and count < 10:
        count += 1
        try:
            # Find videos which are Creative Commons, < 4 minutes, sort by relevance.
            search_ids = youtube_utils.search_youtube(search_text, num_results, "CAASBhABGAEwAQ%253D%253D")
        except Exception as ex:
            break
        video_urls.extend(search_ids)
        if len(video_urls) == 0:
            break

    queue.put((search_tuple, video_urls))


def write_to_file():
    out_file_train = open("youtube_scrape/urls_" + misc_util.get_time_str() + "_relevance_train.csv", "w")
    out_file_val = open("youtube_scrape/urls_" + misc_util.get_time_str() + "_relevance_val.csv", "w")
    vid_ids = {}
    num_add_attempt = 0
    while True:
        search_tuple, urls = queue.get()
        if urls is None:
            break
        if len(urls) == 0:
            continue
        for url in urls:
            num_add_attempt += 1
            if url not in vid_ids:
                vid_ids[url] = []
            vid_ids[url].append(": ".join(search_tuple))
    print("num total", num_add_attempt, "num final", len(vid_ids))
    sorted_vid_ids = sorted(vid_ids.keys())
    random.shuffle(sorted_vid_ids)
    lines_val = sorted_vid_ids[:65536]
    lines_train = sorted_vid_ids[65536:]
    lines_val.sort()
    lines_train.sort()

    lines_train = ['"' + key + '", "' + '", "'.join(vid_ids[key]) + '"' for key in lines_train]
    lines_val = ['"' + key + '", "' + '", "'.join(vid_ids[key]) + '"' for key in lines_val]
    out_file_train.write("\n".join(lines_train) + "\n")
    out_file_train.close()
    out_file_val.write("\n".join(lines_val) + "\n")
    out_file_val.close()


if __name__ == "__main__":
    pool = mp.Pool(num_threads)
    words = [line.strip().split(": ") for line in open("datasets/info_files/full_imagenet_categories_unique.txt")]
    write_proc = mp.Process(target=write_to_file)
    write_proc.daemon = False
    write_proc.start()
    for _ in tqdm.tqdm(pool.imap(search, words), total=len(words)):
        pass

    queue.put((None, None))

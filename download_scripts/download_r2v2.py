import glob
import multiprocessing as mp
import os

import gdown
import tqdm

NUM_THREADS = 8
BASE_DIR = "downloads"


def make_download_url(drive_url):
    return "https://drive.google.com/uc?id=%s" % drive_url.split("?id=")[1]


def download_and_extract(func_args):
    id, filename, out_dir = func_args
    os.makedirs(out_dir, exist_ok=True)
    if len(glob.glob("%s/%s*" % (out_dir, filename))):
        # already downloaded
        return

    url = make_download_url(id)
    download_path = "%s/%s.tar" % (out_dir, filename)
    gdown.download(url, output=download_path)
    if os.path.exists(download_path):
        os.system("tar xf %s -C %s" % (download_path, out_dir))
        os.remove(download_path)


if __name__ == "__main__":
    os.makedirs(BASE_DIR, exist_ok=True)
    os.chdir(BASE_DIR)
    val_url = "https://drive.google.com/open?id=1ixet4jFn1zXRUG5kfwoczFPXjpf7EXFi"
    download_and_extract((val_url, "val", "."))

    pool = mp.Pool(8)
    drive_links = [
        line.strip().split(" ")
        for line in open(
            os.path.join(os.path.dirname(__file__), os.pardir, "datasets", "info_files", "r2v2_drive_urls.txt")
        )
    ]
    filenames, drive_urls = list(zip(*drive_links))
    arg_list = list(zip(drive_urls, filenames, ["train"] * len(filenames)))
    list(tqdm.tqdm(pool.imap(download_and_extract, arg_list), total=len(filenames)))

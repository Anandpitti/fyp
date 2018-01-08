import sys
from icrawler.builtin import GoogleImageCrawler

google_crawler = GoogleImageCrawler(
	parser_threads=5, 
	downloader_threads=10, 
	storage={
		'root_dir': '/home/anandpitti/Desktop/fyp/crawler/dataset/signs_symbols2'
	}
)

google_crawler.crawl(
	keyword='sign symbol in hospital', 
	max_num=1000,
	date_min=None, 
	date_max=None,
	min_size=(200,200), 
	max_size=None
)

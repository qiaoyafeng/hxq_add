import unittest

from common.utils import extract_visual_feature


class MyTestCase(unittest.TestCase):

    def test_extract_visual_feature(self):
        video_path = "10_5_10.mp4"
        extract_visual_feature(video_path)


if __name__ == '__main__':
    unittest.main()

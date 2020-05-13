#
# Created by Qingzhi Ma on Mon May 11 2020
#
# Copyright (c) 2020 Department of Computer Science, University of Warwick
# Copyright 2020 Qingzhi Ma
#
# Licensed under the Apache License, Version 2.0 (the 'License');
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import unittest


class TestSum(unittest.TestCase):
    def test_sum(self):
        data = [1, 2, 3]
        result = sum(data)
        # print("in test 0....")
        self.assertEqual(result, 6)


if __name__ == "__main__":
    unittest.main()

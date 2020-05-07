#
# Created by Qingzhi Ma on Tue May 05 2020
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

from datetime import datetime, timedelta


# convert rfc 3339 time string into a datetime object that represents time in UTC
def to_utc(date_str: str, format_str="%Y-%m-%dt%H:%M:%S.%fz"):
    """ convert rfc 3339 time string into a datetime object that represents time in UTC

    Args:
        date_str (str): date string in rfc 3339 format, like "2015-09-15T17:13:29.380Z"

    Returns:
        datetime: UTC time
    """
    utc_time = datetime.strptime(date_str, format_str)
    return utc_time


def unix_timestamp(date_str: str, format_str="%Y-%m-%dt%H:%M:%S.%fz"):
    """convert rfc 3339 time string into timestamps

    Args:
        date_str (str): date string in rfc 3339 format, like "2015-09-15T17:13:29.380Z"
        format_str (str, optional):  Defaults to "%Y-%m-%dT%H:%M:%S.%fZ".

    Returns:
        float: timestamp
    """
    milliseconds = (to_utc(date_str, format_str) -
                    datetime(1970, 1, 1)) // timedelta(milliseconds=1)
    return milliseconds


if __name__ == "__main__":
    print(unix_timestamp("2019-03-28t16:00:00.000z"))

#!/usr/bin/env python3
"""
Prints the location of a specific Github user
"""


import sys
import requests
from datetime import datetime, timedelta

def get_user_location(api_url):
    """Function to get the location of userf"""
    response = requests.get(api_url)

    if response.status_code == 404:
        print("Not found")
    elif response.status_code == 403:
        reset_time = int(response.headers['X-Ratelimit-Reset'])
        current_time = int(datetime.now().timestamp())
        wait_time = max(0, reset_time - current_time)
        wait_minutes = wait_time // 60
        print(f"Reset in {wait_minutes} min")
    elif response.status_code == 200:
        user_data = response.json()
        location = user_data.get('location')
        if location:
            print(location)
        else:
            print("Location not available")
    else:
        print("Unexpected error")

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: ./2-user_location.py <API_URL>")
        sys.exit(1)

    api_url = sys.argv[1]
    get_user_location(api_url)

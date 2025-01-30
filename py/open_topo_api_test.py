import requests
import sys

def test_opentopo_connection():
    """Test basic connectivity to OpenTopography servers"""
    try:
        # Test connection to main OpenTopography domain
        response = requests.get('https://opentopography.org', timeout=10)
        print(f"\n1. Connection Test:")
        print(f"Status: {'SUCCESS' if response.status_code == 200 else 'FAILED'}")
        print(f"Status code: {response.status_code}")
        return response.status_code == 200
    except requests.exceptions.RequestException as e:
        print(f"\n1. Connection Test FAILED")
        print(f"Error: {str(e)}")
        return False

def test_api_key(api_key):
    """Test if the API key is valid using a minimal API request"""
    # API endpoint for a minimal test request
    url = "https://portal.opentopography.org/API/globaldem"
    
    # Test parameters
    params = {
        'demtype': 'SRTMGL1',
        'south': 36.5,
        'north': 36.6,
        'west': -121.1,
        'east': -121.0,
        'outputFormat': 'GTiff',
        'API_Key': api_key
    }
    
    try:
        print(f"\n2. API Key Test:")
        response = requests.get(url, params=params)
        print(f"Status code: {response.status_code}")
        print(f"Response content: {response.text[:500]}...")  # Print first 500 chars
        return response.status_code == 200
    except requests.exceptions.RequestException as e:
        print(f"API Test FAILED")
        print(f"Error: {str(e)}")
        return False

def check_api_response_format(api_key):
    """Test the API response format"""
    url = "https://portal.opentopography.org/API/globaldem"
    
    # Intentionally invalid parameters to trigger an error response
    params = {
        'demtype': 'INVALID',
        'API_Key': api_key
    }
    
    try:
        print(f"\n3. API Response Format Test:")
        response = requests.get(url, params=params)
        print(f"Status code: {response.status_code}")
        print("Response content:")
        print(response.text)
        
        # Check if response contains error tags
        has_error_tags = '<error>' in response.text
        print(f"\nResponse contains <error> tags: {has_error_tags}")
        
        return has_error_tags
    except requests.exceptions.RequestException as e:
        print(f"Format Test FAILED")
        print(f"Error: {str(e)}")
        return False

if __name__ == "__main__":
    # Your API key
    API_KEY = "dae06e21ac1a1950b0770f05959b3927"
    
    print("OpenTopography API Diagnostic Tests")
    print("==================================")
    
    # Run all tests
    connection_ok = test_opentopo_connection()
    if connection_ok:
        api_ok = test_api_key(API_KEY)
        if api_ok:
            format_ok = check_api_response_format(API_KEY)
    
    print("\nTest Summary:")
    print("============")
    print(f"1. Connection Test: {'✓' if connection_ok else '✗'}")
    if connection_ok:
        print(f"2. API Key Test: {'✓' if api_ok else '✗'}")
        if api_ok:
            print(f"3. Response Format Test: {'✓' if format_ok else '✗'}")
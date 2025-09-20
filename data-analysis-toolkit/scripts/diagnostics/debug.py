#!/usr/bin/env python3
"""
NASA EONET API v2.1 Debug Script
Tests all API endpoints and validates responses
"""

import requests
import json
import sys
from datetime import datetime
from typing import Dict, List, Optional, Any

class EONETAPIDebugger:
    def __init__(self):
        self.base_url = "https://eonet.gsfc.nasa.gov/api/v2.1"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'EONET-API-Debugger/1.0'
        })
        self.results = []
    
    def log_result(self, endpoint: str, status: str, details: str, response_data: Optional[Dict] = None):
        """Log test results"""
        result = {
            'timestamp': datetime.now().isoformat(),
            'endpoint': endpoint,
            'status': status,
            'details': details,
            'response_sample': response_data
        }
        self.results.append(result)
        
        # Color coding for terminal output
        color = {
            'PASS': '\033[92m',    # Green
            'FAIL': '\033[91m',    # Red
            'WARN': '\033[93m',    # Yellow
            'INFO': '\033[94m',    # Blue
            'ENDC': '\033[0m'      # End color
        }
        
        print(f"{color.get(status, '')}{status}: {endpoint}{color['ENDC']}")
        print(f"  Details: {details}")
        if response_data:
            print(f"  Sample: {json.dumps(response_data, indent=2)[:200]}...")
        print()
    
    def test_endpoint(self, url: str, expected_fields: List[str] = None) -> Optional[Dict]:
        """Generic endpoint tester"""
        try:
            response = self.session.get(url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                # Validate expected fields if provided
                if expected_fields and isinstance(data, dict):
                    missing_fields = [field for field in expected_fields if field not in data]
                    if missing_fields:
                        self.log_result(url, 'WARN', 
                                      f"Missing expected fields: {missing_fields}", 
                                      data)
                        return data
                
                self.log_result(url, 'PASS', 
                              f"Status: {response.status_code}, Content-Type: {response.headers.get('content-type')}", 
                              data if isinstance(data, dict) else {'sample': data[:3] if isinstance(data, list) else str(data)[:100]})
                return data
                
            else:
                self.log_result(url, 'FAIL', 
                              f"HTTP {response.status_code}: {response.text[:100]}")
                return None
                
        except requests.exceptions.Timeout:
            self.log_result(url, 'FAIL', "Request timeout")
        except requests.exceptions.ConnectionError:
            self.log_result(url, 'FAIL', "Connection error")
        except json.JSONDecodeError as e:
            self.log_result(url, 'FAIL', f"JSON decode error: {e}")
        except Exception as e:
            self.log_result(url, 'FAIL', f"Unexpected error: {e}")
        
        return None
    
    def test_events_api(self):
        """Test Events API endpoints"""
        print("=" * 60)
        print("TESTING EVENTS API")
        print("=" * 60)
        
        # Basic events endpoint
        events_data = self.test_endpoint(
            f"{self.base_url}/events",
            expected_fields=['title', 'description', 'link', 'events']
        )
        
        # Test with various parameters
        test_params = [
            "?limit=5",
            "?days=20", 
            "?status=open",
            "?status=closed",
            "?source=InciWeb",
            "?limit=5&days=20&status=open",
            "?limit=3&source=InciWeb,EO"
        ]
        
        for params in test_params:
            self.test_endpoint(f"{self.base_url}/events{params}")
        
        # Test individual event if we have event data
        if events_data and 'events' in events_data and events_data['events']:
            event_id = events_data['events'][0].get('id')
            if event_id:
                self.test_endpoint(f"{self.base_url}/events/{event_id}")
    
    def test_categories_api(self):
        """Test Categories API endpoints"""
        print("=" * 60)
        print("TESTING CATEGORIES API") 
        print("=" * 60)
        
        # Basic categories endpoint
        categories_data = self.test_endpoint(
            f"{self.base_url}/categories",
            expected_fields=['title', 'description', 'link', 'categories']
        )
        
        # Test specific category IDs (common ones from documentation)
        category_ids = [8, 14, 6, 9, 10, 12, 13, 15, 16, 17, 18, 19, 20]
        
        for cat_id in category_ids[:3]:  # Test first 3 to avoid too many requests
            self.test_endpoint(f"{self.base_url}/categories/{cat_id}")
            
            # Test with parameters
            self.test_endpoint(f"{self.base_url}/categories/{cat_id}?limit=3")
            self.test_endpoint(f"{self.base_url}/categories/{cat_id}?status=open")
    
    def test_sources_api(self):
        """Test Sources API"""
        print("=" * 60)
        print("TESTING SOURCES API")
        print("=" * 60)
        
        # Sources endpoint
        self.test_endpoint(
            f"{self.base_url}/sources",
            expected_fields=['title', 'description', 'link', 'sources']
        )
    
    def test_layers_api(self):
        """Test Layers API endpoints"""
        print("=" * 60)
        print("TESTING LAYERS API")
        print("=" * 60)
        
        # Basic layers endpoint
        self.test_endpoint(
            f"{self.base_url}/layers",
            expected_fields=['title', 'description', 'link', 'layers']
        )
        
        # Test with category filters
        category_ids = [8, 14]  # Test a couple category filters
        for cat_id in category_ids:
            self.test_endpoint(f"{self.base_url}/layers/{cat_id}")
    
    def validate_event_structure(self, event: Dict) -> List[str]:
        """Validate individual event structure against documentation"""
        issues = []
        required_fields = ['id', 'title', 'link', 'categories', 'sources', 'geometries']
        
        for field in required_fields:
            if field not in event:
                issues.append(f"Missing required field: {field}")
        
        # Check categories structure
        if 'categories' in event:
            if not isinstance(event['categories'], list) or len(event['categories']) == 0:
                issues.append("Categories should be non-empty list")
            else:
                for cat in event['categories']:
                    if not isinstance(cat, dict) or 'id' not in cat or 'title' not in cat:
                        issues.append("Category missing id or title")
        
        # Check geometries structure
        if 'geometries' in event:
            if not isinstance(event['geometries'], list):
                issues.append("Geometries should be a list")
            else:
                for geom in event['geometries']:
                    if 'date' not in geom or 'coordinates' not in geom:
                        issues.append("Geometry missing date or coordinates")
        
        return issues
    
    def run_data_validation(self):
        """Run additional data validation tests"""
        print("=" * 60)
        print("TESTING DATA VALIDATION")
        print("=" * 60)
        
        # Get some events and validate their structure
        try:
            response = self.session.get(f"{self.base_url}/events?limit=5")
            if response.status_code == 200:
                data = response.json()
                if 'events' in data:
                    for i, event in enumerate(data['events'][:3]):  # Check first 3 events
                        issues = self.validate_event_structure(event)
                        if issues:
                            self.log_result(f"Event {i+1} validation", 'WARN', 
                                          f"Structure issues: {'; '.join(issues)}")
                        else:
                            self.log_result(f"Event {i+1} validation", 'PASS', 
                                          "Event structure is valid")
        except Exception as e:
            self.log_result("Data validation", 'FAIL', f"Could not validate data: {e}")
    
    def test_error_conditions(self):
        """Test error handling"""
        print("=" * 60)
        print("TESTING ERROR CONDITIONS")
        print("=" * 60)
        
        # Test invalid endpoints
        error_tests = [
            f"{self.base_url}/events/999999999",  # Non-existent event
            f"{self.base_url}/categories/999999", # Non-existent category
            f"{self.base_url}/invalid_endpoint",  # Invalid endpoint
            f"{self.base_url}/events?limit=abc",  # Invalid parameter
            f"{self.base_url}/events?days=-5"     # Invalid days value
        ]
        
        for url in error_tests:
            try:
                response = self.session.get(url, timeout=5)
                self.log_result(f"Error test: {url.split('/')[-1]}", 'INFO', 
                              f"Status: {response.status_code}, Response: {response.text[:100]}")
            except Exception as e:
                self.log_result(f"Error test: {url}", 'INFO', f"Exception: {e}")
    
    def generate_report(self):
        """Generate summary report"""
        print("=" * 60)
        print("SUMMARY REPORT")
        print("=" * 60)
        
        total_tests = len(self.results)
        passed = len([r for r in self.results if r['status'] == 'PASS'])
        failed = len([r for r in self.results if r['status'] == 'FAIL']) 
        warnings = len([r for r in self.results if r['status'] == 'WARN'])
        
        print(f"Total Tests: {total_tests}")
        print(f"Passed: {passed}")
        print(f"Failed: {failed}")
        print(f"Warnings: {warnings}")
        print(f"Success Rate: {(passed/total_tests)*100:.1f}%")
        
        if failed > 0:
            print("\nFailed Tests:")
            for result in self.results:
                if result['status'] == 'FAIL':
                    print(f"  - {result['endpoint']}: {result['details']}")
        
        # Save detailed results to file
        with open('data_analysis\data-analysis-toolkit\scripts\diagnostics\eonet_debug_results.json', 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"\nDetailed results saved to: eonet_debug_results.json")
    
    def run_all_tests(self):
        """Run all debugging tests"""
        print("NASA EONET API v2.1 Debug Script")
        print(f"Testing API at: {self.base_url}")
        print(f"Started at: {datetime.now().isoformat()}")
        print()
        
        try:
            self.test_events_api()
            self.test_categories_api() 
            self.test_sources_api()
            self.test_layers_api()
            self.run_data_validation()
            self.test_error_conditions()
            
        except KeyboardInterrupt:
            print("\n\nTest interrupted by user")
        except Exception as e:
            print(f"\n\nUnexpected error during testing: {e}")
        finally:
            self.generate_report()

def main():
    """Main function"""
    debugger = EONETAPIDebugger()
    debugger.run_all_tests()

if __name__ == "__main__":
    main()
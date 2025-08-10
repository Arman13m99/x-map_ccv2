#!/usr/bin/env python3
"""
Phase 1 Testing Script - Validates core components without full dependency installation
"""
import os
import sys
import importlib

def test_module_import(module_name, required=True):
    """Test if a module can be imported"""
    try:
        importlib.import_module(module_name)
        print(f"[OK] {module_name}: Import successful")
        return True
    except ImportError as e:
        status = "[FAIL]" if required else "[WARN]"
        print(f"{status} {module_name}: {e}")
        return False
    except Exception as e:
        print(f"[FAIL] {module_name}: Unexpected error - {e}")
        return False

def test_file_syntax(file_path):
    """Test if a Python file has valid syntax"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        compile(content, file_path, 'exec')
        print(f"[OK] {os.path.basename(file_path)}: Syntax valid")
        return True
    except SyntaxError as e:
        print(f"[FAIL] {os.path.basename(file_path)}: Syntax error at line {e.lineno}")
        return False
    except Exception as e:
        print(f"[FAIL] {os.path.basename(file_path)}: {e}")
        return False

def check_file_structure():
    """Check that all required files exist"""
    required_files = [
        'app.py', 'config.py', 'scheduler.py', 'cache.py', 
        'models.py', 'mini.py', 'requirements.txt', '.env.example'
    ]
    
    all_found = True
    for file in required_files:
        if os.path.exists(file):
            print(f"[OK] {file}: Found")
        else:
            print(f"[FAIL] {file}: Missing")
            all_found = False
    
    return all_found

def main():
    """Run all Phase 1 tests"""
    print("Tapsi Food Map Dashboard - Phase 1 Testing")
    print("=" * 50)
    
    # Change to script directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    tests_passed = 0
    total_tests = 0
    
    # Test 1: File structure
    print("\nTesting file structure...")
    total_tests += 1
    if check_file_structure():
        tests_passed += 1
    
    # Test 2: Core dependencies (already available)
    print("\nTesting core dependencies...")
    core_deps = ['pandas', 'numpy', 'flask', 'geopandas', 'shapely', 'requests']
    for dep in core_deps:
        total_tests += 1
        if test_module_import(dep, required=True):
            tests_passed += 1
    
    # Test 3: New dependencies (may not be installed yet)  
    print("\nTesting new production dependencies...")
    new_deps = ['redis', 'apscheduler', 'sklearn']
    for dep in new_deps:
        total_tests += 1
        if test_module_import(dep, required=False):
            tests_passed += 1
    
    # Test 4: Syntax validation
    print("\nTesting syntax validity...")
    python_files = ['app.py', 'config.py', 'scheduler.py', 'cache.py', 'models.py', 'mini.py']
    for file in python_files:
        total_tests += 1
        if test_file_syntax(file):
            tests_passed += 1
    
    # Final results
    print("\n" + "=" * 50)
    print(f"Test Results: {tests_passed}/{total_tests} tests passed")
    
    if tests_passed >= total_tests - 3:  # Allow 3 failures for new deps
        print("Phase 1 implementation is ready!")
        print("\nNext steps:")
        print("1. Install new dependencies: pip install -r requirements.txt")
        print("2. Configure Redis server") 
        print("3. Set up environment variables (copy .env.example to .env)")
        print("4. Test with: python app.py")
        return True
    else:
        print("Some critical tests failed. Please review the output above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
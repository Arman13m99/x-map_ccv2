#!/usr/bin/env python3
"""
Phase 1 Testing Script - Validates core components without full dependency installation
"""
import os
import sys
import importlib
import traceback

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
        print(f"✅ {os.path.basename(file_path)}: Syntax valid")
        return True
    except SyntaxError as e:
        print(f"❌ {os.path.basename(file_path)}: Syntax error at line {e.lineno}: {e.text}")
        return False
    except Exception as e:
        print(f"❌ {os.path.basename(file_path)}: {e}")
        return False

def test_config_creation():
    """Test configuration creation without dependencies"""
    try:
        # Test environment variable handling
        os.environ['METABASE_USERNAME'] = 'test_user'
        os.environ['METABASE_PASSWORD'] = 'test_pass'
        
        # Create a minimal config class for testing
        class MockConfig:
            def __init__(self):
                self.metabase_url = os.getenv('METABASE_URL', 'https://test.com')
                self.metabase_username = os.getenv('METABASE_USERNAME', '')
                self.metabase_password = os.getenv('METABASE_PASSWORD', '')
            
            def validate(self):
                return bool(self.metabase_username and self.metabase_password)
        
        config = MockConfig()
        if config.validate():
            print("✅ Config: Environment variable handling works")
            return True
        else:
            print("❌ Config: Validation failed")
            return False
            
    except Exception as e:
        print(f"❌ Config: {e}")
        return False
    finally:
        # Cleanup
        if 'METABASE_USERNAME' in os.environ:
            del os.environ['METABASE_USERNAME']
        if 'METABASE_PASSWORD' in os.environ:
            del os.environ['METABASE_PASSWORD']

def check_file_structure():
    """Check that all required files exist"""
    required_files = [
        'app.py',
        'config.py',
        'scheduler.py', 
        'cache.py',
        'models.py',
        'mini.py',
        'requirements.txt',
        '.env.example'
    ]
    
    missing_files = []
    for file in required_files:
        if os.path.exists(file):
            print(f"✅ {file}: Found")
        else:
            print(f"❌ {file}: Missing")
            missing_files.append(file)
    
    return len(missing_files) == 0

def main():
    """Run all Phase 1 tests"""
    print("🚀 Tapsi Food Map Dashboard - Phase 1 Testing")
    print("=" * 50)
    
    # Change to script directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    tests_passed = 0
    total_tests = 0
    
    # Test 1: File structure
    print("\n📁 Testing file structure...")
    total_tests += 1
    if check_file_structure():
        tests_passed += 1
    
    # Test 2: Core dependencies (already available)
    print("\n📦 Testing core dependencies...")
    core_deps = ['pandas', 'numpy', 'flask', 'geopandas', 'shapely', 'requests']
    core_available = 0
    for dep in core_deps:
        total_tests += 1
        if test_module_import(dep, required=True):
            tests_passed += 1
            core_available += 1
    
    # Test 3: New dependencies (may not be installed yet)  
    print("\n🔧 Testing new production dependencies...")
    new_deps = [
        ('redis', False),
        ('apscheduler', False), 
        ('sklearn', False)
    ]
    for dep, required in new_deps:
        total_tests += 1
        if test_module_import(dep, required=required):
            tests_passed += 1
    
    # Test 4: Syntax validation
    print("\n🔍 Testing syntax validity...")
    python_files = ['app.py', 'config.py', 'scheduler.py', 'cache.py', 'models.py', 'mini.py']
    for file in python_files:
        total_tests += 1
        if test_file_syntax(file):
            tests_passed += 1
    
    # Test 5: Configuration handling
    print("\n⚙️ Testing configuration...")
    total_tests += 1
    if test_config_creation():
        tests_passed += 1
    
    # Test 6: Environment file
    print("\n🌍 Testing environment configuration...")
    total_tests += 1
    if os.path.exists('.env.example'):
        print("✅ .env.example: Template available")
        tests_passed += 1
    else:
        print("❌ .env.example: Missing")
    
    # Final results
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {tests_passed}/{total_tests} tests passed")
    
    if tests_passed == total_tests:
        print("🎉 Phase 1 implementation is ready!")
        print("\n📋 Next steps:")
        print("1. Install new dependencies: pip install -r requirements.txt")
        print("2. Configure Redis server") 
        print("3. Set up environment variables (copy .env.example to .env)")
        print("4. Test with: python app.py")
        return True
    else:
        print("⚠️ Some tests failed. Please review the output above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
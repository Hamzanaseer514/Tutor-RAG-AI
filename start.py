#!/usr/bin/env python3
"""
Tuterby AI Startup Script
Run this to start the Tuterby AI application
"""

import os
import sys
import uvicorn
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def check_environment():
    """Check if required environment variables are set"""
    required_vars = ['GROK_API_KEY']
    missing_vars = []
    
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        print("❌ Missing required environment variables:")
        for var in missing_vars:
            print(f"   - {var}")
        print("\nPlease create a .env file with the required variables.")
        print("See env_template.txt for reference.")
        return False
    
    print("✅ Environment variables configured")
    return True

def main():
    """Main startup function"""
    print("🚀 Starting Tuterby AI...")
    
    # Check environment
    if not check_environment():
        sys.exit(1)
    
    # Start the application
    print("🌐 Starting web server on http://localhost:8000")
    print("📚 Tuterby AI is ready to process your PDFs!")
    print("\nPress Ctrl+C to stop the server")
    
    try:
        uvicorn.run(
            "app.main:app",
            host="0.0.0.0",
            port=8000,
            reload=True,
            log_level="info"
        )
    except KeyboardInterrupt:
        print("\n👋 Tuterby AI stopped. Goodbye!")
    except Exception as e:
        print(f"❌ Error starting Tuterby AI: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

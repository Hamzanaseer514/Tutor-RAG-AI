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
    # Require Gemini API key
    if not os.getenv('GEMINI_API_KEY'):
        print("❌ Missing required environment variables:")
        print("   - GEMINI_API_KEY")
        print("\nPlease create a .env file with your Gemini API key.")
        print("See env_template.txt for reference (GEMINI_API_KEY=...)")
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

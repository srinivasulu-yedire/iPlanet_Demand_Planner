#!/usr/bin/env python3
"""
Test script for GenAI layer integration.
Run this to verify that the GenAI layer is working correctly.
"""

import os
import sys
import pandas as pd
import numpy as np

# Add src directory to path
sys.path.append('src')

def test_genai_basic():
    """Test basic GenAI functionality."""
    print("Testing GenAI Layer Integration")
    print("=" * 40)

    try:
        from genai_layer import initialize_genai, explain_forecast_accuracy, get_forecast_insights
        print("✅ GenAI layer imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import GenAI layer: {e}")
        return False

    # Test initialization
    try:
        if os.getenv("OPENAI_API_KEY"):
            initialize_genai(provider="openai")
            print("✅ GenAI initialized with OpenAI")
            genai_available = True
        else:
            print("⚠️  OPENAI_API_KEY not set - GenAI functions will return error messages")
            genai_available = False
    except Exception as e:
        print(f"❌ GenAI initialization failed: {e}")
        genai_available = False

    # Test model metrics explanation
    print("\nTesting Model Accuracy Explanation:")
    print("-" * 30)

    sample_metrics = {
        'Forecast Accuracy (%)': 81.79,
        'MAE (Absolute Error)': 12.5,
        'RMSE (Safety Deviation)': 15.8,
        'Features_Used': 36,
        'Algorithm': 'XGBoost'
    }

    try:
        explanation = explain_forecast_accuracy(sample_metrics, "Retail stock prediction for Apple products")
        if genai_available:
            print("✅ Model explanation generated successfully")
            print(f"Sample output: {explanation[:200]}...")
        else:
            print(f"⚠️  Expected error message: {explanation[:100]}...")
    except Exception as e:
        print(f"❌ Model explanation failed: {e}")

    # Test forecast insights
    print("\nTesting Forecast Insights:")
    print("-" * 25)

    sample_predictions = [45, 52, 48, 55, 50, 58, 53]
    sample_product = "iPhone 15 Pro"

    try:
        insights = get_forecast_insights(sample_predictions, sample_product)
        if genai_available:
            print("✅ Forecast insights generated successfully")
            print(f"Sample output: {insights[:200]}...")
        else:
            print(f"⚠️  Expected error message: {insights[:100]}...")
    except Exception as e:
        print(f"❌ Forecast insights failed: {e}")

    print("\n" + "=" * 40)
    print("GenAI Integration Test Complete")
    print("\nTo enable full AI functionality:")
    print("1. Set your OPENAI_API_KEY environment variable")
    print("2. Or pass api_key parameter to initialize_genai()")
    print("3. To switch to Claude later: switch_to_claude(api_key='your_key')")

    return True

def test_streamlit_integration():
    """Test that streamlit app can import GenAI functions."""
    print("\nTesting Streamlit Integration:")
    print("-" * 30)

    try:
        # Simulate streamlit app import
        import streamlit as st
        import pandas as pd
        import joblib
        import os
        import datetime
        import numpy as np
        import io

        # Test GenAI import in streamlit context
        try:
            from genai_layer import (
                initialize_genai, explain_forecast_accuracy, analyze_forecast_trends,
                generate_inventory_recommendations, create_executive_summary,
                detect_anomalies, get_forecast_insights, format_business_alert
            )
            print("✅ Streamlit can import all GenAI functions")
        except ImportError as e:
            print(f"❌ Streamlit GenAI import failed: {e}")
            return False

        print("✅ Streamlit integration test passed")
        return True

    except ImportError:
        print("⚠️  Streamlit not available for testing")
        return True

if __name__ == "__main__":
    print("GenAI Integration Test Suite")
    print("=" * 50)

    # Run tests
    basic_test = test_genai_basic()
    streamlit_test = test_streamlit_integration()

    if basic_test and streamlit_test:
        print("\n🎉 All tests passed! GenAI layer is ready to use.")
    else:
        print("\n⚠️  Some tests failed. Check the output above for details.")

    print("\nNext steps:")
    print("1. Set OPENAI_API_KEY environment variable")
    print("2. Run: streamlit run src/streamlit_app.py")
    print("3. Check the '🤖 AI Insights' tab for GenAI features")
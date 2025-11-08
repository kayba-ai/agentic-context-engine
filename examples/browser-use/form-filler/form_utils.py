#!/usr/bin/env python3
"""
Form-filler specific utilities.

Functions specific to form filling examples.
"""

from typing import List, Dict


def get_test_forms() -> List[Dict]:
    """
    Get list of test forms to fill.

    Returns:
        List of form definitions with name and data fields
    """
    return [
        {
            "name": "Contact Form",
            "data": {
                "name": "John Doe",
                "email": "john@example.com",
                "message": "Hello, this is a test message.",
            },
        },
        {
            "name": "Newsletter Signup",
            "data": {"email": "jane@example.com", "name": "Jane Smith"},
        },
        {
            "name": "User Registration",
            "data": {
                "username": "testuser123",
                "email": "test@example.com",
                "password": "SecurePass123",
            },
        },
    ]

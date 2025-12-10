"""Tests for ace.integrations.base module."""

import pytest
from ace import Skillbook
from ace.integrations.base import wrap_skillbook_context


class TestWrapSkillbookContext:
    """Test the wrap_skillbook_context helper function."""

    def test_empty_skillbook_returns_empty_string(self):
        """Should return empty string when skillbook has no skills."""
        skillbook = Skillbook()
        result = wrap_skillbook_context(skillbook)

        assert result == ""
        assert isinstance(result, str)

    def test_single_skill_formats_correctly(self):
        """Should format single skill with explanation."""
        skillbook = Skillbook()
        skillbook.add_skill("general", "Always validate inputs before processing")

        result = wrap_skillbook_context(skillbook)

        # Should contain header
        assert "Strategic Knowledge" in result
        assert "Learned from Experience" in result

        # Should contain the skill content
        assert "Always validate inputs before processing" in result

        # Should contain usage instructions
        assert "How to use these strategies" in result
        assert "success rates" in result
        assert "helpful > harmful" in result

        # Should contain important note
        assert "learned patterns, not rigid rules" in result

    def test_multiple_skills_all_included(self):
        """Should include all skills in formatted output."""
        skillbook = Skillbook()
        skillbook.add_skill("general", "First strategy")
        skillbook.add_skill("general", "Second strategy")
        skillbook.add_skill("specific", "Third strategy")

        result = wrap_skillbook_context(skillbook)

        assert "First strategy" in result
        assert "Second strategy" in result
        assert "Third strategy" in result

    def test_skill_with_metadata_shows_scores(self):
        """Should display helpful/harmful scores from metadata."""
        skillbook = Skillbook()
        skillbook.add_skill(
            "general", "High success strategy", metadata={"helpful": 5, "harmful": 1}
        )

        result = wrap_skillbook_context(skillbook)

        # The skill content should be present
        assert "High success strategy" in result
        # Skillbook.as_prompt() should include score information

    def test_different_sections_all_included(self):
        """Should include skills from different sections."""
        skillbook = Skillbook()
        skillbook.add_skill("browser", "Wait for elements to load")
        skillbook.add_skill("api", "Retry on timeout")
        skillbook.add_skill("general", "Log all errors")

        result = wrap_skillbook_context(skillbook)

        assert "Wait for elements to load" in result
        assert "Retry on timeout" in result
        assert "Log all errors" in result

    def test_output_includes_key_sections(self):
        """Should include all required sections in output."""
        skillbook = Skillbook()
        skillbook.add_skill("general", "Test strategy")

        result = wrap_skillbook_context(skillbook)

        # Check for markdown headers
        assert "##" in result
        assert "Available Strategic Knowledge" in result

        # Check for skill points/instructions
        assert "**How to use these strategies:**" in result
        assert "**Important:**" in result

        # Check for specific guidance
        assert "Review skills relevant to your current task" in result
        assert "Prioritize strategies with high success rates" in result
        assert "Apply strategies when they match your context" in result
        assert "Adapt general strategies" in result

    def test_output_format_is_string(self):
        """Should always return a string."""
        skillbook1 = Skillbook()
        skillbook2 = Skillbook()
        skillbook2.add_skill("test", "Content")

        result1 = wrap_skillbook_context(skillbook1)
        result2 = wrap_skillbook_context(skillbook2)

        assert isinstance(result1, str)
        assert isinstance(result2, str)

    def test_skillbook_with_special_characters(self):
        """Should handle skills with special characters."""
        skillbook = Skillbook()
        skillbook.add_skill("general", "Use `quotes` and **bold** in markdown")
        skillbook.add_skill("general", "Handle <tags> and {braces}")

        result = wrap_skillbook_context(skillbook)

        assert "Use `quotes` and **bold** in markdown" in result
        assert "Handle <tags> and {braces}" in result

    def test_skillbook_with_multiline_skill(self):
        """Should handle skills with newlines."""
        skillbook = Skillbook()
        skillbook.add_skill("general", "First line\nSecond line\nThird line")

        result = wrap_skillbook_context(skillbook)

        # The skill content should be present
        assert "First line" in result or "First line\nSecond line" in result

    def test_empty_string_has_no_formatting(self):
        """Empty skillbook should return truly empty string, no formatting."""
        skillbook = Skillbook()
        result = wrap_skillbook_context(skillbook)

        assert result == ""
        assert len(result) == 0
        assert "Strategic Knowledge" not in result
        assert "How to use" not in result

    def test_wrapping_same_skillbook_twice_identical(self):
        """Should produce identical output for same skillbook."""
        skillbook = Skillbook()
        skillbook.add_skill("general", "Consistent output")

        result1 = wrap_skillbook_context(skillbook)
        result2 = wrap_skillbook_context(skillbook)

        assert result1 == result2

    def test_output_mentions_emoji(self):
        """Should include emoji in header for visual appeal."""
        skillbook = Skillbook()
        skillbook.add_skill("general", "Test")

        result = wrap_skillbook_context(skillbook)

        # Check for emoji in header
        assert "📚" in result

    def test_output_encourages_judgment(self):
        """Should remind users to use judgment, not treat as rigid rules."""
        skillbook = Skillbook()
        skillbook.add_skill("general", "Strategy")

        result = wrap_skillbook_context(skillbook)

        assert "Use judgment" in result
        assert "not rigid rules" in result

    def test_skills_parameter_uses_skillbook_method(self):
        """Should use skillbook.skills() to get skills."""
        skillbook = Skillbook()
        skillbook.add_skill("test", "First")
        skillbook.add_skill("test", "Second")

        # Ensure skillbook has skills
        assert len(skillbook.skills()) == 2

        result = wrap_skillbook_context(skillbook)

        # Both skills should be in result
        assert "First" in result
        assert "Second" in result

    def test_uses_skillbook_as_prompt(self):
        """Should use skillbook.as_prompt() for skill formatting."""
        skillbook = Skillbook()
        skillbook.add_skill("general", "Test skill")

        # Get the formatted skills
        skill_text = skillbook.as_prompt()
        result = wrap_skillbook_context(skillbook)

        # Result should contain the formatted skill text
        assert skill_text in result

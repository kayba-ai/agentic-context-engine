"""
Tool Selection Task Environment for ACE Training

Evaluates the quality of tool selection decisions made by an AI assistant
based on conversation traces from Helicone logs.
"""

from typing import Dict, Any, Optional
from dataclasses import dataclass

from ace import TaskEnvironment, EnvironmentResult
from helicone_loader import HeliconeTrace, ConversationTurn, ToolCall


@dataclass
class ToolSelectionSample:
    """A sample representing a tool selection decision point"""

    turn_index: int
    system_prompt: str
    conversation_history: list[ConversationTurn]
    user_request: str
    selected_tool: str
    tool_input: Dict[str, Any]
    tool_result: Optional[Dict[str, Any]]
    next_user_feedback: Optional[str]  # What the user said after this action


class ToolSelectionEnvironment(TaskEnvironment):
    """
    Environment for evaluating tool selection decisions.

    This environment assesses whether an agent made good tool choices
    based on:
    1. Whether the tool succeeded or failed
    2. Whether it matched the pattern in successful traces
    3. Whether the user gave positive/negative feedback afterward
    4. Efficiency metrics (tokens, cost, latency)
    """

    def __init__(self, trace: HeliconeTrace):
        """
        Initialize with a complete conversation trace.

        Args:
            trace: HeliconeTrace containing the full conversation
        """
        self.trace = trace
        self.samples = self._extract_samples()
        self.current_sample_idx = 0

    def _extract_samples(self) -> list[ToolSelectionSample]:
        """Extract tool selection decision points from the trace"""
        samples = []

        for i, turn in enumerate(self.trace.conversation):
            if turn.role != "assistant" or not turn.tool_calls:
                continue

            # Get conversation history up to this point
            history = [
                t for t in self.trace.conversation if t.turn_index < turn.turn_index
            ]

            # Get the most recent user request
            user_request = ""
            for hist_turn in reversed(history):
                if hist_turn.role == "user" and hist_turn.text_content:
                    user_request = hist_turn.text_content[0]
                    break

            # Get next user feedback (if any)
            next_user_feedback = None
            if i + 1 < len(self.trace.conversation):
                next_turn = self.trace.conversation[i + 1]
                if next_turn.role == "user" and next_turn.text_content:
                    next_user_feedback = next_turn.text_content[0]

            # Create sample for each tool call in this turn
            for tool_call in turn.tool_calls:
                sample = ToolSelectionSample(
                    turn_index=turn.turn_index,
                    system_prompt=self.trace.system_prompt,
                    conversation_history=history,
                    user_request=user_request,
                    selected_tool=tool_call.tool_name,
                    tool_input=tool_call.tool_input,
                    tool_result=tool_call.tool_result,
                    next_user_feedback=next_user_feedback,
                )
                samples.append(sample)

        return samples

    def evaluate(self, answer: str) -> EnvironmentResult:
        """
        Evaluate a tool selection decision.

        Args:
            answer: The agent's response describing which tool to use and why

        Returns:
            EnvironmentResult with feedback on the tool selection
        """
        if self.current_sample_idx >= len(self.samples):
            return EnvironmentResult(
                feedback="No more samples to evaluate", correct=False
            )

        sample = self.samples[self.current_sample_idx]

        # Parse the agent's answer to extract tool choice
        predicted_tool = self._extract_tool_from_answer(answer)

        # Evaluate the decision
        is_correct = predicted_tool == sample.selected_tool

        # Generate detailed feedback
        feedback = self._generate_feedback(sample, predicted_tool, is_correct)

        # Ground truth is the actual tool used in the trace
        ground_truth = f"Tool: {sample.selected_tool}, Input: {sample.tool_input}"

        self.current_sample_idx += 1

        return EnvironmentResult(
            feedback=feedback, correct=is_correct, ground_truth=ground_truth
        )

    def _extract_tool_from_answer(self, answer: str) -> str:
        """Extract the tool name from the agent's response"""
        # Simple extraction - look for tool names in the answer
        known_tools = [
            "writeFile",
            "readFile",
            "edit_file",
            "deleteFile",
            "get_codebase_context",
            "code_check",
            "install",
        ]

        answer_lower = answer.lower()

        for tool in known_tools:
            if tool.lower() in answer_lower:
                return tool

        return "unknown"

    def _generate_feedback(
        self, sample: ToolSelectionSample, predicted_tool: str, is_correct: bool
    ) -> str:
        """Generate detailed feedback for the tool selection"""

        if is_correct:
            feedback = f"✓ CORRECT: Selected '{predicted_tool}' which matches the actual trace.\n"
        else:
            feedback = f"✗ INCORRECT: Selected '{predicted_tool}' but actual trace used '{sample.selected_tool}'.\n"

        # Add context
        feedback += f"\nContext:\n"
        feedback += f"  - Turn: {sample.turn_index}\n"
        feedback += f"  - User request: {sample.user_request[:100]}...\n"
        feedback += f"  - Previous tools: {self._get_previous_tools(sample)}\n"

        # Add reasoning about why this tool made sense
        feedback += f"\nAnalysis:\n"
        feedback += self._analyze_tool_choice(sample)

        # Add user feedback if available
        if sample.next_user_feedback:
            feedback += f"\nUser's next message: {sample.next_user_feedback[:150]}...\n"
            if self._indicates_success(sample.next_user_feedback):
                feedback += "  → User seems satisfied (positive signal)\n"
            elif self._indicates_correction(sample.next_user_feedback):
                feedback += (
                    "  → User seems to be correcting/clarifying (negative signal)\n"
                )

        return feedback

    def _get_previous_tools(self, sample: ToolSelectionSample) -> list[str]:
        """Get list of tools used before this decision"""
        tools = []
        for turn in sample.conversation_history:
            for tool_call in turn.tool_calls:
                tools.append(tool_call.tool_name)
        return tools

    def _analyze_tool_choice(self, sample: ToolSelectionSample) -> str:
        """Provide analysis of why this tool choice makes sense"""

        tool = sample.selected_tool
        analysis = []

        # Pattern-based reasoning
        if tool == "get_codebase_context":
            analysis.append(
                "  - get_codebase_context is appropriate at the start to understand the project"
            )

        elif tool == "writeFile":
            analysis.append("  - writeFile is used for creating new files")
            prev_tools = self._get_previous_tools(sample)
            if "writeFile" in prev_tools:
                analysis.append("  - Continuing file creation pattern from earlier")

        elif tool == "edit_file":
            analysis.append("  - edit_file is for modifying existing files")
            analysis.append("  - Typically used after initial file creation")

        elif tool == "readFile":
            analysis.append("  - readFile is used to inspect existing file contents")

        elif tool == "code_check":
            analysis.append("  - code_check validates the code for errors")
            prev_tools = self._get_previous_tools(sample)
            write_count = prev_tools.count("writeFile") + prev_tools.count("edit_file")
            if write_count >= 3:
                analysis.append(
                    f"  - Good practice to check after {write_count} file operations"
                )

        elif tool == "install":
            analysis.append("  - install adds new dependencies to the project")

        return (
            "\n".join(analysis)
            if analysis
            else "  - Standard tool choice for this situation"
        )

    def _indicates_success(self, feedback: str) -> bool:
        """Check if user feedback indicates success"""
        positive_signals = [
            "great",
            "good",
            "perfect",
            "excellent",
            "thanks",
            "working",
            "looks good",
            "that works",
        ]
        feedback_lower = feedback.lower()
        return any(signal in feedback_lower for signal in positive_signals)

    def _indicates_correction(self, feedback: str) -> bool:
        """Check if user feedback indicates a correction"""
        negative_signals = [
            "actually",
            "no",
            "wrong",
            "instead",
            "fix",
            "error",
            "issue",
            "problem",
            "not",
        ]
        feedback_lower = feedback.lower()
        return any(signal in feedback_lower for signal in negative_signals)

    def get_num_samples(self) -> int:
        """Return the number of samples in this environment"""
        return len(self.samples)

    def reset(self):
        """Reset the environment to the beginning"""
        self.current_sample_idx = 0

    def format_sample_as_question(self, sample_idx: int) -> str:
        """
        Format a sample as a question for the Agent to answer.

        Args:
            sample_idx: Index of the sample to format

        Returns:
            A string question describing the context and asking for tool selection
        """
        if sample_idx >= len(self.samples):
            return "No more samples available"

        sample = self.samples[sample_idx]

        question = "You are an AI coding assistant. Based on the following context, "
        question += "which tool should you use next?\n\n"

        question += f"**User Request:**\n{sample.user_request}\n\n"

        prev_tools = self._get_previous_tools(sample)
        if prev_tools:
            question += f"**Previous Tools Used:**\n"
            question += ", ".join(prev_tools) + "\n\n"

        question += "**Available Tools:**\n"
        question += "- get_codebase_context: Understand the project structure\n"
        question += "- writeFile: Create a new file\n"
        question += "- readFile: Read an existing file\n"
        question += "- edit_file: Modify an existing file\n"
        question += "- deleteFile: Delete a file\n"
        question += "- code_check: Validate code for errors\n"
        question += "- install: Add dependencies\n\n"

        question += "**Question:** Which tool should you use next and why?\n"
        question += "Respond with: TOOL: <tool_name> | REASONING: <your reasoning>"

        return question


def main():
    """Demo: Create environment and show samples"""
    import os
    from helicone_loader import HeliconeLoader

    # Data files are stored in .private/helicone/ to keep them out of the repo
    # Place your Helicone JSON export in: ../../.private/helicone/oneline.json
    script_dir = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(script_dir, "../../.private/helicone/oneline.json")

    # Load trace
    loader = HeliconeLoader(json_path)
    trace = loader.load()

    # Create environment
    env = ToolSelectionEnvironment(trace)

    print("=" * 80)
    print("TOOL SELECTION ENVIRONMENT")
    print("=" * 80)
    print(f"\nTotal samples: {env.get_num_samples()}\n")

    # Show first 3 samples
    for i in range(min(3, env.get_num_samples())):
        print(f"\n{'=' * 80}")
        print(f"SAMPLE {i + 1}")
        print("=" * 80)

        question = env.format_sample_as_question(i)
        print(question)

        sample = env.samples[i]
        print(f"\n**Ground Truth:**")
        print(f"  Actual tool used: {sample.selected_tool}")
        print(f"  Tool input: {sample.tool_input}")

        # Simulate evaluation
        simulated_answer = f"I would use {sample.selected_tool} because it's appropriate for this context."
        result = env.evaluate(simulated_answer)

        print(f"\n**Evaluation Result:**")
        print(f"  Correct: {result.correct}")
        print(f"\n**Feedback:**")
        print(result.feedback)


if __name__ == "__main__":
    main()

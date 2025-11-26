"""
Execution Trace Parser for Claude Code Output

Transforms unstructured Claude Code output into structured execution traces
similar to browseruse format, enabling the Reflector to extract concrete
implementation patterns instead of vague philosophy.
"""

import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field


@dataclass
class ExecutionStep:
    """Represents a single execution step in Claude Code's work."""
    step_num: int
    description: str
    memory: str  # What the agent was thinking/reasoning
    action: str  # Tool name and parameters
    result: str  # Outcome of the action
    pattern: str = ""  # Identified implementation pattern
    code_changes: str = ""  # Summary of code changes if applicable


@dataclass
class ImplementationPattern:
    """Represents a concrete implementation pattern found in the code."""
    type: str  # e.g., "class_creation", "interface_definition", "serialization"
    description: str
    evidence: str  # Where in the code this appears
    line_count: int = 0


class ExecutionTraceParser:
    """Parse Claude Code output into structured execution traces."""

    def __init__(self):
        # Tool invocation patterns
        self.tool_patterns = {
            'Read': [
                r'Reading (?:file )?([^\n]+)',
                r'I\'ll (?:now )?read ([^\n]+)',
                r'Let me (?:first )?read ([^\n]+)',
                r'Examining ([^\n]+)',
                r'Looking at ([^\n]+)',
            ],
            'Write': [
                r'Writing (?:to )?([^\n]+)',
                r'Creating (?:new )?(?:file )?([^\n]+)',
                r'I\'ll (?:now )?(?:create|write) ([^\n]+)',
                r'Let me (?:create|write) ([^\n]+)',
            ],
            'Edit': [
                r'Editing ([^\n]+)',
                r'Updating ([^\n]+)',
                r'Modifying ([^\n]+)',
                r'I\'ll (?:now )?(?:edit|update|modify) ([^\n]+)',
            ],
            'Bash': [
                r'Running (?:command)?:? (.+)',
                r'Executing:? (.+)',
                r'I\'ll (?:now )?run (.+)',
                r'Running `([^`]+)`',
            ],
            'Test': [
                r'Running tests',
                r'Executing test suite',
                r'npm (?:run )?test',
            ]
        }

        # Reasoning/decision patterns
        self.reasoning_patterns = [
            r'(?:First|Next|Now|Then),? I(?:\'ll| will| need to) (.+)',
            r'The (.+) (?:needs?|requires?) (.+)',
            r'I (?:need to|should|will) (.+)',
            r'Let me (.+)',
            r'(?:My|The) (?:plan|approach|strategy) is to (.+)',
            r'Based on (.+), I(?:\'ll| will) (.+)',
            r'Since (.+), (?:I|we) (.+)',
            r'To (.+), I(?:\'ll| will) (.+)',
        ]

        # Result/outcome patterns
        self.result_patterns = [
            r'✓ (.+)',
            r'✅ (.+)',
            r'Successfully (.+)',
            r'Created (.+)',
            r'Updated (.+)',
            r'Found (.+)',
            r'Identified (.+)',
            r'Completed (.+)',
            r'Fixed (.+)',
            r'Added (.+)',
            r'Implemented (.+)',
        ]

        # Section markers
        self.section_markers = [
            r'^#{1,3} (.+)$',  # Markdown headers
            r'^(?:Step|Task) \d+:? (.+)',
            r'^\d+\. (.+)',  # Numbered lists
        ]

    def parse_claude_code_output(self, stdout: str) -> List[ExecutionStep]:
        """
        Parse Claude Code's output into structured execution steps.

        Looks for the new structured format:
        --- Step N: Description ---
        Reasoning: ...
        Action: Tool(params)
        Result: ...
        Pattern: ...

        Args:
            stdout: Raw output from Claude Code

        Returns:
            List of ExecutionStep objects representing the work done
        """
        steps = []
        lines = stdout.split('\n')

        # Pattern to match step headers: --- Step N: Description ---
        step_header_pattern = r'^---\s*Step\s+(\d+):\s*(.+?)\s*---$'

        i = 0
        while i < len(lines):
            line = lines[i].strip()

            # Check for step header
            header_match = re.match(step_header_pattern, line, re.IGNORECASE)
            if header_match:
                step_num = int(header_match.group(1))
                description = header_match.group(2)

                # Extract the step details (Reasoning, Action, Result, Pattern)
                reasoning = ""
                action = ""
                result = ""
                pattern = ""

                # Look at next few lines for step details
                j = i + 1
                while j < len(lines) and j < i + 10:  # Look ahead max 10 lines
                    detail_line = lines[j].strip()

                    # Stop at next step header
                    if re.match(step_header_pattern, detail_line, re.IGNORECASE):
                        break

                    # Extract fields
                    if detail_line.startswith('Reasoning:'):
                        reasoning = detail_line[10:].strip()
                    elif detail_line.startswith('Action:'):
                        action = detail_line[7:].strip()
                    elif detail_line.startswith('Result:'):
                        result = detail_line[7:].strip()
                    elif detail_line.startswith('Pattern:'):
                        pattern = detail_line[8:].strip()

                    j += 1

                # Create step if we have at least action
                if action:
                    steps.append(ExecutionStep(
                        step_num=step_num,
                        description=description,
                        memory=reasoning if reasoning else "Continuing translation work",
                        action=action,
                        result=result if result else "(processing...)",
                        pattern=pattern if pattern else "Implementation step"
                    ))

                i = j  # Skip to where we left off
            else:
                i += 1

        # Fallback: If no structured steps found, try old parsing method
        if not steps:
            print("⚠️  No structured execution trace found, falling back to heuristic parsing")
            steps = self._parse_legacy_format(stdout)

        return steps

    def _parse_legacy_format(self, stdout: str) -> List[ExecutionStep]:
        """
        Fallback parser for when structured format isn't present.
        Uses heuristic extraction from sections and headers.
        """
        steps = []
        lines = stdout.split('\n')
        current_step = None
        current_reasoning = []
        step_counter = 1

        i = 0
        while i < len(lines):
            line = lines[i].strip()

            # Skip empty lines
            if not line:
                i += 1
                continue

            # Check for tool invocations
            tool_found = False
            for tool_name, patterns in self.tool_patterns.items():
                for pattern in patterns:
                    match = re.search(pattern, line, re.IGNORECASE)
                    if match:
                        # Save previous step if exists
                        if current_step:
                            steps.append(current_step)

                        # Extract target/parameters
                        target = match.group(1) if match.groups() else tool_name

                        # Look back for reasoning
                        reasoning = self._extract_preceding_reasoning(lines, i, current_reasoning)

                        # Create new step
                        current_step = ExecutionStep(
                            step_num=step_counter,
                            description=self._generate_step_description(tool_name, target),
                            memory=reasoning,
                            action=f"{tool_name}('{target}')",
                            result="(processing...)"
                        )
                        step_counter += 1
                        current_reasoning = []  # Reset reasoning buffer
                        tool_found = True
                        break

                if tool_found:
                    break

            # If no tool found, check if this is reasoning/decision
            if not tool_found:
                for pattern in self.reasoning_patterns:
                    if re.search(pattern, line, re.IGNORECASE):
                        current_reasoning.append(line)
                        break

                # Check for results if we have a current step
                if current_step:
                    for pattern in self.result_patterns:
                        match = re.search(pattern, line, re.IGNORECASE)
                        if match:
                            result_text = match.group(1) if match.groups() else line
                            current_step.result = result_text
                            break

            i += 1

        # Don't forget the last step
        if current_step:
            steps.append(current_step)

        # If still no steps found, create high-level steps from sections
        if not steps:
            steps = self._extract_steps_from_sections(stdout)

        return steps

    def _extract_preceding_reasoning(self, lines: List[str], current_index: int,
                                    reasoning_buffer: List[str]) -> str:
        """Extract reasoning that precedes a tool invocation."""
        # Use reasoning buffer if available
        if reasoning_buffer:
            return ' '.join(reasoning_buffer)

        # Otherwise look back up to 5 lines
        reasoning_lines = []
        for j in range(max(0, current_index - 5), current_index):
            line = lines[j].strip()
            if line and not line.startswith('#'):
                # Check if this line contains reasoning
                for pattern in self.reasoning_patterns:
                    if re.search(pattern, line, re.IGNORECASE):
                        reasoning_lines.append(line)
                        break

        return ' '.join(reasoning_lines) if reasoning_lines else "Continuing translation work"

    def _generate_step_description(self, tool: str, target: str) -> str:
        """Generate a descriptive title for the step."""
        if tool == 'Read':
            if 'source' in target.lower():
                return "Analyze Python Source"
            elif 'test' in target.lower():
                return "Review Test Files"
            else:
                return f"Read {self._extract_filename(target)}"
        elif tool == 'Write':
            if 'test' in target.lower():
                return "Create Test Suite"
            elif '.ts' in target:
                return f"Create TypeScript {self._extract_filename(target)}"
            else:
                return f"Write {self._extract_filename(target)}"
        elif tool == 'Edit':
            return f"Update {self._extract_filename(target)}"
        elif tool == 'Bash':
            if 'test' in target:
                return "Run Tests"
            elif 'build' in target or 'compile' in target:
                return "Build Project"
            else:
                return "Execute Command"
        else:
            return f"{tool} Operation"

    def _extract_filename(self, path: str) -> str:
        """Extract just the filename from a path."""
        # Remove workspace prefix
        path = re.sub(r'^workspace/', '', path)
        # Get last component
        parts = path.split('/')
        return parts[-1] if parts else path

    def _extract_steps_from_sections(self, stdout: str) -> List[ExecutionStep]:
        """Fallback: Extract steps from section headers if no tool invocations found."""
        steps = []
        lines = stdout.split('\n')
        step_counter = 1

        for i, line in enumerate(lines):
            for pattern in self.section_markers:
                match = re.match(pattern, line.strip())
                if match:
                    section_title = match.group(1)
                    # Get next few lines as content
                    content_lines = []
                    for j in range(i+1, min(i+6, len(lines))):
                        if lines[j].strip() and not re.match(pattern, lines[j]):
                            content_lines.append(lines[j].strip())

                    steps.append(ExecutionStep(
                        step_num=step_counter,
                        description=section_title,
                        memory="Section from Claude Code output",
                        action="Processing",
                        result=' '.join(content_lines)[:200]
                    ))
                    step_counter += 1
                    break

        return steps

    def extract_patterns_from_diff(self, git_diff: str) -> List[ImplementationPattern]:
        """
        Extract concrete implementation patterns from git diff.

        Args:
            git_diff: Raw git diff output

        Returns:
            List of identified implementation patterns
        """
        patterns = []

        # TypeScript class pattern
        class_matches = re.findall(r'\+export class (\w+)', git_diff)
        for class_name in class_matches:
            patterns.append(ImplementationPattern(
                type="class_creation",
                description=f"TypeScript class {class_name}",
                evidence=f"export class {class_name}",
                line_count=self._count_class_lines(git_diff, class_name)
            ))

        # Interface pattern
        interface_matches = re.findall(r'\+(?:export )?interface (\w+)', git_diff)
        for interface_name in interface_matches:
            patterns.append(ImplementationPattern(
                type="interface_definition",
                description=f"TypeScript interface {interface_name}",
                evidence=f"interface {interface_name}",
                line_count=self._count_interface_lines(git_diff, interface_name)
            ))

        # Type alias pattern
        type_matches = re.findall(r'\+(?:export )?type (\w+)', git_diff)
        for type_name in type_matches:
            patterns.append(ImplementationPattern(
                type="type_alias",
                description=f"Type alias {type_name}",
                evidence=f"type {type_name}"
            ))

        # JSON serialization pattern
        if 'toJSON' in git_diff and 'fromJSON' in git_diff:
            patterns.append(ImplementationPattern(
                type="json_serialization",
                description="Bidirectional JSON conversion (toJSON/fromJSON)",
                evidence="toJSON() and fromJSON() methods"
            ))

        # Constructor with defaults pattern
        constructor_match = re.search(r'\+\s*constructor\([^)]+=[^)]+\)', git_diff)
        if constructor_match:
            patterns.append(ImplementationPattern(
                type="constructor_defaults",
                description="Constructor with default parameter values",
                evidence="constructor with = default values"
            ))

        # Strict TypeScript patterns
        if re.search(r'\+\s*"strict":\s*true', git_diff):
            patterns.append(ImplementationPattern(
                type="strict_typescript",
                description="Strict TypeScript configuration enabled",
                evidence='"strict": true in tsconfig'
            ))

        # Test patterns
        test_patterns = [
            (r'\+\s*describe\([\'"]([^"\']+)', "Test suite"),
            (r'\+\s*it\([\'"]([^"\']+)', "Test case"),
            (r'\+\s*expect\(', "Test assertion"),
        ]

        for pattern, desc in test_patterns:
            matches = re.findall(pattern, git_diff)
            if matches:
                patterns.append(ImplementationPattern(
                    type="testing",
                    description=f"{desc}: {len(matches)} instances",
                    evidence=f"Found {len(matches)} {desc.lower()}s"
                ))

        # Union type pattern
        union_matches = re.findall(r'\+\s*\w+:\s*[\'"]?\w+[\'"]?\s*\|\s*[\'"]?\w+', git_diff)
        if union_matches:
            patterns.append(ImplementationPattern(
                type="union_types",
                description=f"Union types for type constraints ({len(union_matches)} uses)",
                evidence="Type unions with | operator"
            ))

        # Generic type pattern
        generic_matches = re.findall(r'\+.*<(\w+)(?:,\s*\w+)*>', git_diff)
        if generic_matches:
            patterns.append(ImplementationPattern(
                type="generics",
                description=f"Generic types ({len(generic_matches)} uses)",
                evidence="Type parameters with <T>"
            ))

        return patterns

    def _count_class_lines(self, diff: str, class_name: str) -> int:
        """Count lines added for a class."""
        in_class = False
        count = 0
        brace_depth = 0

        for line in diff.split('\n'):
            if f'class {class_name}' in line:
                in_class = True

            if in_class and line.startswith('+'):
                count += 1
                brace_depth += line.count('{') - line.count('}')
                if brace_depth == 0 and count > 1:
                    break

        return count

    def _count_interface_lines(self, diff: str, interface_name: str) -> int:
        """Count lines added for an interface."""
        in_interface = False
        count = 0

        for line in diff.split('\n'):
            if f'interface {interface_name}' in line:
                in_interface = True

            if in_interface and line.startswith('+'):
                count += 1
                if '}' in line:
                    break

        return count

    def format_execution_trace(self, steps: List[ExecutionStep],
                              patterns: List[ImplementationPattern],
                              commit_msg: str, success: bool) -> str:
        """
        Format execution steps as browseruse-style traces.

        Args:
            steps: List of execution steps
            patterns: Implementation patterns found
            commit_msg: Git commit message
            success: Whether execution succeeded

        Returns:
            Formatted execution trace string
        """
        trace = "=== CLAUDE CODE EXECUTION TRACE (Chronological) ===\n\n"

        # Add execution steps
        for step in steps:
            trace += f"--- Step {step.step_num}: {step.description} ---\n"
            trace += f"   Memory: {step.memory}\n"
            trace += f"▶️  Action: {step.action}\n"
            trace += f"📊 Result: {step.result}\n"

            if step.code_changes:
                trace += f"📝 Changes: {step.code_changes}\n"

            if step.pattern:
                trace += f"🎯 Pattern: {step.pattern}\n"

            trace += "\n"

        trace += "=== END EXECUTION TRACE ===\n\n"

        # Add implementation patterns section
        if patterns:
            trace += "## IMPLEMENTATION PATTERNS OBSERVED\n\n"
            for i, pattern in enumerate(patterns, 1):
                trace += f"{i}. {pattern.description}\n"
                trace += f"   - Type: {pattern.type}\n"
                trace += f"   - Evidence: {pattern.evidence}\n"
                if pattern.line_count > 0:
                    trace += f"   - Size: {pattern.line_count} lines\n"
                trace += "\n"

        # Add final outcome
        trace += f"## FINAL OUTCOME\n"
        trace += f"Status: {'SUCCESS' if success else 'FAILED'}\n"

        # Extract strategies from commit message
        strategy_pattern = r'Applied strategies:\s*([^\n]+)'
        strategy_match = re.search(strategy_pattern, commit_msg, re.IGNORECASE)
        if strategy_match:
            trace += f"Strategies Acknowledged: {strategy_match.group(1)}\n"

        return trace

    def parse_debug_output(self, debug_output: str, stdout: str) -> List[ExecutionStep]:
        """
        Parse Claude Code's debug output to extract real tool calls.

        Debug output contains lines like:
        [DEBUG] executePreToolHooks called for tool: Bash
        [DEBUG] executePreToolHooks called for tool: Read
        [DEBUG] executePreToolHooks called for tool: Write

        Args:
            debug_output: stderr from Claude Code with --debug flag
            stdout: stdout for matching results to tools

        Returns:
            List of ExecutionStep objects from actual tool calls
        """
        steps = []
        tool_pattern = r'\[DEBUG\] executePreToolHooks called for tool: (\w+)'

        # Extract all tool calls in order
        tool_calls = []
        for line in debug_output.split('\n'):
            match = re.search(tool_pattern, line)
            if match:
                tool_name = match.group(1)
                tool_calls.append(tool_name)

        if not tool_calls:
            print("⚠️  No tool calls found in debug output, falling back to stdout parsing")
            return self.parse_claude_code_output(stdout)

        print(f"✅ Found {len(tool_calls)} real tool calls in debug output: {', '.join(tool_calls)}")

        # Build steps from tool calls
        for i, tool_name in enumerate(tool_calls, 1):
            # Extract actual reasoning from stdout (text before <function_calls>)
            reasoning = self._extract_reasoning_before_tool_call(stdout, i)

            # Find result for this tool call
            _, result = self._find_tool_context_in_stdout(tool_name, i, stdout)

            step = ExecutionStep(
                step_num=i,
                description=self._generate_step_description(tool_name, reasoning),
                memory=reasoning,  # Now contains actual Claude Code reasoning
                action=f"{tool_name}()",
                result=result,
                pattern="Implementation step"  # Will be refined later
            )
            steps.append(step)

        return steps

    def _extract_reasoning_before_tool_call(self, stdout: str, tool_occurrence: int) -> str:
        """
        Extract Claude Code's reasoning that appears before a specific tool call.

        Claude Code outputs reasoning as plain text before <function_calls> blocks.
        We extract the text between function_calls blocks or before the first one.

        Args:
            stdout: Full Claude Code output
            tool_occurrence: Which tool call to find reasoning for (1-indexed)

        Returns:
            The reasoning text, or generic fallback
        """
        # Find all <function_calls> blocks (using DOTALL flag to match across lines)
        blocks = list(re.finditer(r'<function_calls>.*?</function_calls>', stdout, re.DOTALL))

        if not blocks or tool_occurrence > len(blocks):
            return "Preparing to execute task"

        # Get the function_calls block for this occurrence
        target_block = blocks[tool_occurrence - 1]

        # Extract text BEFORE this block
        if tool_occurrence == 1:
            # For first tool call, get everything before the first <function_calls>
            text_before = stdout[:target_block.start()]
        else:
            # For subsequent calls, get text between previous </function_calls> and current <function_calls>
            prev_block = blocks[tool_occurrence - 2]
            text_before = stdout[prev_block.end():target_block.start()]

        # Clean up the reasoning text
        reasoning = text_before.strip()

        # Remove common prefixes/formatting
        reasoning = re.sub(r'^[\s\n]*', '', reasoning)  # Leading whitespace
        reasoning = re.sub(r'</function_results>[\s\n]*', '', reasoning)  # Function results tags
        reasoning = re.sub(r'<system-reminder>.*?</system-reminder>', '', reasoning, flags=re.DOTALL)  # System reminders

        # Take first substantial paragraph (up to 500 chars)
        if len(reasoning) > 500:
            # Find a good break point (end of sentence)
            break_point = reasoning.rfind('.', 0, 500)
            if break_point > 100:
                reasoning = reasoning[:break_point + 1]
            else:
                reasoning = reasoning[:500] + "..."

        # If we got something substantial, return it
        if len(reasoning.strip()) > 20:
            return reasoning.strip()

        # Fallback
        return "Continuing with implementation"

    def _find_tool_context_in_stdout(self, tool_name: str, step_num: int, stdout: str) -> Tuple[str, str]:
        """
        Try to find reasoning and results for a tool call in stdout.

        Args:
            tool_name: Name of the tool (Read, Write, Bash, etc.)
            step_num: Step number for context
            stdout: Full stdout to search

        Returns:
            Tuple of (reasoning, result)
        """
        # Simple heuristic: look for tool-related keywords near step mentions
        lines = stdout.split('\n')
        reasoning = f"Using {tool_name} tool"
        result = "(completed)"

        # Tool-specific context hints
        if tool_name == 'Read':
            for line in lines:
                if any(word in line.lower() for word in ['reading', 'examining', 'analyzing', 'found']):
                    if len(line.strip()) > 10 and len(line.strip()) < 200:
                        result = line.strip()[:150]
                        break
        elif tool_name == 'Write':
            for line in lines:
                if any(word in line.lower() for word in ['created', 'writing', 'added', 'interface', 'class']):
                    if len(line.strip()) > 10 and len(line.strip()) < 200:
                        result = line.strip()[:150]
                        break
        elif tool_name == 'Edit':
            for line in lines:
                if any(word in line.lower() for word in ['updated', 'modified', 'editing', 'changed']):
                    if len(line.strip()) > 10 and len(line.strip()) < 200:
                        result = line.strip()[:150]
                        break
        elif tool_name == 'Bash':
            for line in lines:
                if any(word in line.lower() for word in ['test', 'npm', 'build', 'passed', 'failed']):
                    if len(line.strip()) > 10 and len(line.strip()) < 200:
                        result = line.strip()[:150]
                        break

        return reasoning, result

    def identify_pattern_for_step(self, step: ExecutionStep,
                                 patterns: List[ImplementationPattern]) -> str:
        """Identify which pattern a step contributes to."""
        action_lower = step.action.lower()

        if 'read' in action_lower and 'source' in action_lower:
            return "Source analysis for translation planning"
        elif 'write' in action_lower:
            if 'test' in action_lower:
                return "Test creation pattern"
            elif 'interface' in step.result.lower():
                return "Interface-first design pattern"
            elif 'class' in step.result.lower():
                return "Class implementation pattern"
        elif 'edit' in action_lower:
            if 'json' in step.result.lower():
                return "JSON serialization pattern"
            elif 'constructor' in step.result.lower():
                return "Constructor pattern"
        elif 'bash' in action_lower:
            if 'test' in action_lower:
                return "Test execution pattern"
            elif 'build' in action_lower or 'compile' in action_lower:
                return "Build verification pattern"

        return "Implementation step"


def parse_and_format_execution(stdout: str, git_diff: str, commit_msg: str,
                              success: bool = True, debug_output: str = "") -> str:
    """
    Convenience function to parse and format Claude Code output.

    Args:
        stdout: Claude Code's stdout
        git_diff: Git diff of changes
        commit_msg: Commit message
        success: Whether execution succeeded
        debug_output: Claude Code's debug stderr output

    Returns:
        Formatted execution trace
    """
    parser = ExecutionTraceParser()

    # Parse execution steps from debug output if available
    # NOTE: Claude Code with --print sends debug to both stdout and stderr
    debug_source = debug_output if debug_output else stdout

    if debug_source and '[DEBUG] executePreToolHooks' in debug_source:
        print(f"🐛 Found debug tool traces in output")
        steps = parser.parse_debug_output(debug_source, stdout)
    else:
        print("⚠️  No debug tool traces found, falling back to stdout parsing")
        # Fallback to parsing stdout
        steps = parser.parse_claude_code_output(stdout)

    # Extract patterns from diff
    patterns = parser.extract_patterns_from_diff(git_diff)

    # Identify patterns for each step
    for step in steps:
        step.pattern = parser.identify_pattern_for_step(step, patterns)

    # Format as trace
    return parser.format_execution_trace(steps, patterns, commit_msg, success)
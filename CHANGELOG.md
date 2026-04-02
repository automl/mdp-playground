## [1.0.0] - 2026-04-02

Release Highlights: Version 1.0.0 (Generated using an LLM: Google Gemini)
🚀 Major Breaking Changes & Core Updates
Changed to manage project with uv - change minimum Python version to 3.11.

Improved README.

Gymnasium Migration: Full migration from gym to gymnasium (v1.0.0 compatibility). This includes updated return values for step() and reset() (support for terminated/truncated flags).

API Refactor: Significant renaming of internal functions for clarity, specifically around Markov state management (get_augmented_state, etc.) and image representation.

Dependency Modernization: Upgraded numpy and random number generation to align with modern Gymnasium standards (_np_random).

🛠 Environment Enhancements (RLToyEnv)
Advanced Rendering: Added a more flexible render() function that allows for custom trajectory rollouts and "imaginary" rollouts from specific starting states.

Observation Capabilities: * Improved get_image_representation to support uncertainty visualization (epistemic and aleatoric) via bar plots.

Added support for setting custom dtype_s and dtype_o for state and observation spaces.

Dynamics & Noise: Transition and reward noises can now be state-and-action dependent. Improved default noise profiles for continuous environments.

Reward Logic: Improved reward_every_n_steps logic to work across discrete, continuous, and grid environments.

🧪 Wrappers & Compatibility
Gymnasium Wrapper: Updated wrapper to support irrelevant dimensions and image transformations.

External Integration: Improved support and examples for MiniGrid, ProcGen, and Mujoco (v4) environments.

Resource Management: Added close() functionality to properly release Pygame resources.

📈 Tooling & Documentation
Example Suite: Overhauled example.py with a better CLI, individual function calls, and logging toggles for image observations.

Experimentation: Updated experiment configuration scripts and cleaned up Jupyter notebooks for plotting results.

CI/CD: Updated GitHub workflows to support newer Python versions and fixed code coverage reporting.

🐛 Bug Fixes
Fixed issues with copy.deepcopy() by removing redundant state variables (self.P, self.R).

Resolved reward bugs related to delays exceeding sequence lengths.

Fixed terminal state logic for grid environments.

Rectified various test failures in TestGymEnvWrapper and TestRLToyEnv.

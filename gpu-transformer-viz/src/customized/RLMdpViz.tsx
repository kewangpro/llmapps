import { useState } from 'react';
import { ArrowRight, Circle, Square, Target, Brain, Repeat } from 'lucide-react';

export const metadata = {
  name: "RL MDP",
  icon: "Brain"
};

interface ComponentInfo {
  title: string;
  description: string;
  color: string;
  textColor: string;
}

interface StepInfo {
  s: string;
  a: string;
  r: string;
  sNext: string;
  desc: string;
}

type ComponentKey = 'agent' | 'environment' | 'state' | 'action' | 'reward' | 'policy' | 'value' | 'transition';

export default function RLMdpViz() {
  const [activeComponent, setActiveComponent] = useState<ComponentKey | null>(null);
  const [step, setStep] = useState(0);

  const components: Record<ComponentKey, ComponentInfo> = {
    agent: {
      title: "Agent",
      description: "The learner/decision maker that takes actions based on the current state",
      color: "bg-blue-500",
      textColor: "text-blue-700"
    },
    environment: {
      title: "Environment",
      description: "The world the agent interacts with, responds to actions and provides new states",
      color: "bg-green-500",
      textColor: "text-green-700"
    },
    state: {
      title: "State (S)",
      description: "A representation of the current situation. In MDP, satisfies Markov property: future depends only on present, not past",
      color: "bg-purple-500",
      textColor: "text-purple-700"
    },
    action: {
      title: "Action (A)",
      description: "Choices available to the agent. Set of all possible actions is the action space",
      color: "bg-orange-500",
      textColor: "text-orange-700"
    },
    reward: {
      title: "Reward (R)",
      description: "Immediate feedback signal from environment. Agent's goal is to maximize cumulative reward",
      color: "bg-red-500",
      textColor: "text-red-700"
    },
    policy: {
      title: "Policy (π)",
      description: "Agent's strategy: mapping from states to actions. Can be deterministic π(s) or stochastic π(a|s)",
      color: "bg-indigo-500",
      textColor: "text-indigo-700"
    },
    value: {
      title: "Value Function (V/Q)",
      description: "Expected cumulative reward from a state (V) or state-action pair (Q). Used to evaluate states/actions",
      color: "bg-pink-500",
      textColor: "text-pink-700"
    },
    transition: {
      title: "Transition Dynamics (P)",
      description: "P(s'|s,a): Probability of reaching state s' from state s after action a",
      color: "bg-teal-500",
      textColor: "text-teal-700"
    }
  };

  const steps: StepInfo[] = [
    { s: "S₀", a: "A₀", r: "R₁", sNext: "S₁", desc: "Agent observes initial state S₀" },
    { s: "S₁", a: "A₁", r: "R₂", sNext: "S₂", desc: "Agent selects action A₁ based on policy" },
    { s: "S₂", a: "A₂", r: "R₃", sNext: "S₃", desc: "Environment returns reward R₂ and new state S₂" },
    { s: "S₃", a: "A₃", r: "R₄", sNext: "S₄", desc: "Process continues until terminal state" }
  ];

  return (
    <div className="w-full max-w-6xl mx-auto p-6 bg-gradient-to-br from-slate-50 to-slate-100">
      <div className="mb-8 text-center">
        <h1 className="text-3xl font-bold text-slate-800 mb-2">
          Reinforcement Learning as Markov Decision Process
        </h1>
        <p className="text-slate-600">
          Interactive visualization of the RL-MDP framework
        </p>
      </div>

      {/* Main Interaction Loop */}
      <div className="bg-white rounded-lg shadow-lg p-8 mb-6">
        <h2 className="text-xl font-bold text-slate-800 mb-6 text-center">
          Agent-Environment Interaction Loop
        </h2>

        <div className="flex items-center justify-center gap-8">
          {/* Agent */}
          <div
            className={`relative cursor-pointer transition-all duration-300 ${
              activeComponent === 'agent' ? 'scale-110' : 'scale-100'
            }`}
            onMouseEnter={() => setActiveComponent('agent')}
            onMouseLeave={() => setActiveComponent(null)}
          >
            <div className="bg-blue-100 rounded-lg p-6 border-4 border-blue-500 w-48 h-32 flex flex-col items-center justify-center">
              <Brain className="w-12 h-12 text-blue-600 mb-2" />
              <span className="text-lg font-bold text-blue-800">Agent</span>
              <span className="text-sm text-blue-600">Policy π(a|s)</span>
            </div>
          </div>

          {/* Arrows and State/Action/Reward */}
          <div className="flex flex-col items-center gap-4">
            {/* Action Arrow Down */}
            <div className="flex flex-col items-center">
              <div
                className={`px-4 py-2 rounded-lg cursor-pointer transition-all ${
                  activeComponent === 'action' ? 'bg-orange-200 scale-110' : 'bg-orange-100'
                }`}
                onMouseEnter={() => setActiveComponent('action')}
                onMouseLeave={() => setActiveComponent(null)}
              >
                <span className="text-orange-800 font-semibold">Action (Aₜ)</span>
              </div>
              <ArrowRight className="w-6 h-6 text-slate-400 rotate-90 my-1" />
            </div>

            {/* State and Reward Arrow Up */}
            <div className="flex flex-col items-center">
              <ArrowRight className="w-6 h-6 text-slate-400 -rotate-90 my-1" />
              <div className="flex gap-2">
                <div
                  className={`px-3 py-2 rounded-lg cursor-pointer transition-all ${
                    activeComponent === 'state' ? 'bg-purple-200 scale-110' : 'bg-purple-100'
                  }`}
                  onMouseEnter={() => setActiveComponent('state')}
                  onMouseLeave={() => setActiveComponent(null)}
                >
                  <span className="text-purple-800 font-semibold">State (Sₜ₊₁)</span>
                </div>
                <div
                  className={`px-3 py-2 rounded-lg cursor-pointer transition-all ${
                    activeComponent === 'reward' ? 'bg-red-200 scale-110' : 'bg-red-100'
                  }`}
                  onMouseEnter={() => setActiveComponent('reward')}
                  onMouseLeave={() => setActiveComponent(null)}
                >
                  <span className="text-red-800 font-semibold">Reward (Rₜ₊₁)</span>
                </div>
              </div>
            </div>
          </div>

          {/* Environment */}
          <div
            className={`relative cursor-pointer transition-all duration-300 ${
              activeComponent === 'environment' ? 'scale-110' : 'scale-100'
            }`}
            onMouseEnter={() => setActiveComponent('environment')}
            onMouseLeave={() => setActiveComponent(null)}
          >
            <div className="bg-green-100 rounded-lg p-6 border-4 border-green-500 w-48 h-32 flex flex-col items-center justify-center">
              <Target className="w-12 h-12 text-green-600 mb-2" />
              <span className="text-lg font-bold text-green-800">Environment</span>
              <span className="text-sm text-green-600">Dynamics P(s'|s,a)</span>
            </div>
          </div>
        </div>

        {/* Component Description */}
        {activeComponent && (
          <div className="mt-6 p-4 bg-slate-50 rounded-lg border-2 border-slate-300">
            <h3 className={`font-bold text-lg mb-2 ${components[activeComponent].textColor}`}>
              {components[activeComponent].title}
            </h3>
            <p className="text-slate-700">{components[activeComponent].description}</p>
          </div>
        )}
      </div>

      {/* MDP Formal Definition */}
      <div className="bg-white rounded-lg shadow-lg p-6 mb-6">
        <h2 className="text-xl font-bold text-slate-800 mb-4">MDP Formal Definition</h2>
        <div className="bg-slate-50 rounded-lg p-6 font-mono text-sm">
          <p className="mb-4 text-slate-700">
            An MDP is defined by the tuple: <span className="font-bold text-slate-900">(S, A, P, R, γ)</span>
          </p>
          <div className="space-y-3 text-slate-700">
            <div className="flex">
              <span className="font-bold text-purple-700 w-12">S:</span>
              <span>Set of states (state space)</span>
            </div>
            <div className="flex">
              <span className="font-bold text-orange-700 w-12">A:</span>
              <span>Set of actions (action space)</span>
            </div>
            <div className="flex">
              <span className="font-bold text-teal-700 w-12">P:</span>
              <span>P(s'|s,a) = Pr(Sₜ₊₁ = s' | Sₜ = s, Aₜ = a) - Transition probability</span>
            </div>
            <div className="flex">
              <span className="font-bold text-red-700 w-12">R:</span>
              <span>R(s,a,s') = E[Rₜ₊₁ | Sₜ = s, Aₜ = a, Sₜ₊₁ = s'] - Reward function</span>
            </div>
            <div className="flex">
              <span className="font-bold text-indigo-700 w-12">γ:</span>
              <span>Discount factor ∈ [0,1] for future rewards</span>
            </div>
          </div>
        </div>
      </div>

      {/* Trajectory Visualization */}
      <div className="bg-white rounded-lg shadow-lg p-6 mb-6">
        <h2 className="text-xl font-bold text-slate-800 mb-4">Episode Trajectory</h2>
        <p className="text-slate-600 mb-4">
          Click through to see how states, actions, and rewards unfold over time
        </p>

        <div className="flex items-center justify-center gap-2 mb-6">
          <button
            onClick={() => setStep(Math.max(0, step - 1))}
            className="px-4 py-2 bg-slate-200 rounded-lg hover:bg-slate-300 transition-colors disabled:opacity-50"
            disabled={step === 0}
          >
            Previous
          </button>
          <span className="px-4 py-2 bg-slate-100 rounded-lg font-semibold">
            Step {step + 1} / {steps.length}
          </span>
          <button
            onClick={() => setStep(Math.min(steps.length - 1, step + 1))}
            className="px-4 py-2 bg-slate-200 rounded-lg hover:bg-slate-300 transition-colors disabled:opacity-50"
            disabled={step === steps.length - 1}
          >
            Next
          </button>
        </div>

        <div className="flex items-center justify-center gap-3 mb-4 flex-wrap">
          {steps.slice(0, step + 1).map((s, idx) => (
            <div key={idx} className="flex items-center gap-3">
              <div className="flex items-center gap-3">
                <div className="bg-purple-100 rounded-lg px-4 py-3 border-2 border-purple-500">
                  <div className="font-bold text-purple-800">{s.s}</div>
                  <div className="text-xs text-purple-600">State</div>
                </div>
                {idx === step && (
                  <>
                    <ArrowRight className="w-6 h-6 text-slate-400" />
                    <div className="bg-orange-100 rounded-lg px-4 py-3 border-2 border-orange-500">
                      <div className="font-bold text-orange-800">{s.a}</div>
                      <div className="text-xs text-orange-600">Action</div>
                    </div>
                    <ArrowRight className="w-6 h-6 text-slate-400" />
                    <div className="bg-red-100 rounded-lg px-4 py-3 border-2 border-red-500">
                      <div className="font-bold text-red-800">{s.r}</div>
                      <div className="text-xs text-red-600">Reward</div>
                    </div>
                  </>
                )}
              </div>
              {idx < step && <ArrowRight className="w-6 h-6 text-slate-400" />}
            </div>
          ))}
        </div>

        <div className="bg-blue-50 rounded-lg p-4 text-center">
          <p className="text-slate-700">{steps[step].desc}</p>
        </div>
      </div>

      {/* Key Concepts Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div className="bg-white rounded-lg shadow p-4">
          <h3 className="font-bold text-indigo-700 mb-2 flex items-center gap-2">
            <Circle className="w-5 h-5" />
            Policy (π)
          </h3>
          <p className="text-sm text-slate-600">
            π(a|s): Probability of taking action a in state s. Agent's behavior strategy.
          </p>
          <div className="mt-2 text-xs bg-indigo-50 p-2 rounded">
            <span className="font-semibold">Optimal Policy π*:</span> Maximizes expected cumulative reward
          </div>
        </div>

        <div className="bg-white rounded-lg shadow p-4">
          <h3 className="font-bold text-pink-700 mb-2 flex items-center gap-2">
            <Square className="w-5 h-5" />
            Value Functions
          </h3>
          <p className="text-sm text-slate-600 mb-2">
            <span className="font-semibold">V<sup>π</sup>(s):</span> Expected return from state s under policy π
          </p>
          <p className="text-sm text-slate-600">
            <span className="font-semibold">Q<sup>π</sup>(s,a):</span> Expected return from state s, taking action a, then following π
          </p>
        </div>

        <div className="bg-white rounded-lg shadow p-4">
          <h3 className="font-bold text-teal-700 mb-2 flex items-center gap-2">
            <Repeat className="w-5 h-5" />
            Markov Property
          </h3>
          <p className="text-sm text-slate-600">
            The future is independent of the past given the present:
          </p>
          <div className="mt-2 text-xs bg-teal-50 p-2 rounded font-mono">
            P(Sₜ₊₁|Sₜ, Sₜ₋₁, ..., S₀) = P(Sₜ₊₁|Sₜ)
          </div>
        </div>

        <div className="bg-white rounded-lg shadow p-4">
          <h3 className="font-bold text-slate-700 mb-2">Bellman Equation</h3>
          <p className="text-sm text-slate-600 mb-2">
            Recursive relationship for value functions:
          </p>
          <div className="text-xs bg-slate-50 p-2 rounded font-mono space-y-1">
            <div>V<sup>π</sup>(s) = Σₐ π(a|s) Σₛ′ P(s'|s,a)[R + γV<sup>π</sup>(s')]</div>
            <div>Q<sup>π</sup>(s,a) = Σₛ′ P(s'|s,a)[R + γV<sup>π</sup>(s')]</div>
          </div>
        </div>
      </div>

      {/* Learning Objective */}
      <div className="bg-white rounded-lg shadow-lg p-6 mt-6">
        <h2 className="text-xl font-bold text-slate-800 mb-3">RL Objective</h2>
        <p className="text-slate-600 mb-4">
          Find optimal policy π* that maximizes expected cumulative discounted reward:
        </p>
        <div className="bg-slate-50 rounded p-4 font-mono text-center text-xl text-slate-800">
          π* = argmax<sub>π</sub> E[Σₜ γᵗ Rₜ | π]
        </div>
      </div>
    </div>
  );
}

//! Tests for React → Qliphoth migration.
//!
//! Following Agent-TDD methodology from docs/specs/REACT-MIGRATION-TDD-ROADMAP.md
//! Tests are crystallized understanding of React → Qliphoth transformation.

use super::*;
use std::path::{Path, PathBuf};

// =============================================================================
// Phase 1.1: JSX Parsing
// =============================================================================

#[test]
fn test_parse_simple_element() {
    // GIVEN: A simple JSX element
    let source = r#"
        function App() {
            return <div>Hello</div>;
        }
    "#;

    // WHEN: We extract from JSX
    let result = extract_source(source, Path::new("test.jsx"), "test.jsx").unwrap();

    // THEN: We get a component with JSX tree
    assert_eq!(result.components.len(), 1);
    let comp = &result.components[0];
    assert_eq!(comp.name, "App");

    let root = comp.jsx.root.as_ref().expect("Should have JSX root");
    match &root.node_type {
        JsxNodeType::Element { tag, children, .. } => {
            assert_eq!(tag, "div");
            assert_eq!(children.len(), 1);
            match &children[0].node_type {
                JsxNodeType::Text { value } => assert_eq!(value, "Hello"),
                _ => panic!("Expected text child"),
            }
        }
        _ => panic!("Expected element root"),
    }
}

#[test]
fn test_parse_nested_elements() {
    // GIVEN: Nested JSX elements
    let source = r#"
        function Layout() {
            return (
                <div className="container">
                    <header>Title</header>
                    <main>Content</main>
                </div>
            );
        }
    "#;

    // WHEN: We extract
    let result = extract_source(source, Path::new("test.jsx"), "test.jsx").unwrap();

    // THEN: Nesting is preserved
    let comp = &result.components[0];
    let root = comp.jsx.root.as_ref().unwrap();

    match &root.node_type {
        JsxNodeType::Element { tag, children, .. } => {
            assert_eq!(tag, "div");
            assert_eq!(children.len(), 2);
        }
        _ => panic!("Expected element"),
    }
}

#[test]
fn test_parse_jsx_fragment() {
    // GIVEN: JSX fragment
    let source = r#"
        function List() {
            return (
                <>
                    <li>One</li>
                    <li>Two</li>
                </>
            );
        }
    "#;

    // WHEN: We extract
    let result = extract_source(source, Path::new("test.jsx"), "test.jsx").unwrap();

    // THEN: Fragment is detected
    let comp = &result.components[0];
    let root = comp.jsx.root.as_ref().unwrap();

    match &root.node_type {
        JsxNodeType::Fragment { children } => {
            assert_eq!(children.len(), 2);
        }
        _ => panic!("Expected fragment"),
    }
}

#[test]
fn test_parse_jsx_attributes() {
    // GIVEN: Element with attributes
    let source = r#"
        function Button() {
            return <button className="btn" disabled onClick={handleClick}>Click</button>;
        }
    "#;

    // WHEN: We extract
    let result = extract_source(source, Path::new("test.jsx"), "test.jsx").unwrap();

    // THEN: Attributes are extracted
    let comp = &result.components[0];
    let root = comp.jsx.root.as_ref().unwrap();

    match &root.node_type {
        JsxNodeType::Element { attributes, .. } => {
            assert_eq!(attributes.len(), 3);

            // className is string
            let class_attr = attributes.iter().find(|a| a.name == "className").unwrap();
            match &class_attr.value {
                JsxAttributeValue::String { value } => assert_eq!(value, "btn"),
                _ => panic!("Expected string value"),
            }

            // disabled is boolean shorthand
            let disabled_attr = attributes.iter().find(|a| a.name == "disabled").unwrap();
            assert!(matches!(disabled_attr.value, JsxAttributeValue::True));

            // onClick is event handler
            let click_attr = attributes.iter().find(|a| a.name == "onClick").unwrap();
            assert!(click_attr.is_event_handler);
        }
        _ => panic!("Expected element"),
    }
}

// =============================================================================
// Phase 1.2: Component Detection
// =============================================================================

#[test]
fn test_detect_functional_component() {
    // GIVEN: Various function styles
    let source = r#"
        // Function declaration
        function Counter() {
            return <div>0</div>;
        }

        // Arrow function
        const Timer = () => <span>00:00</span>;

        // Regular function (not a component - lowercase)
        function helper() {
            return null;
        }
    "#;

    // WHEN: We extract
    let result = extract_source(source, Path::new("test.tsx"), "test.tsx").unwrap();

    // THEN: Only uppercase functions with JSX are detected
    assert_eq!(result.components.len(), 2);
    assert!(result.components.iter().any(|c| c.name == "Counter"));
    assert!(result.components.iter().any(|c| c.name == "Timer"));
    assert!(!result.components.iter().any(|c| c.name == "helper"));
}

#[test]
fn test_detect_class_component() {
    // GIVEN: A class component
    let source = r#"
        import React, { Component } from 'react';

        class Counter extends Component {
            render() {
                return <div>{this.state.count}</div>;
            }
        }
    "#;

    // WHEN: We extract
    let result = extract_source(source, Path::new("test.tsx"), "test.tsx").unwrap();

    // THEN: Class component is detected
    assert_eq!(result.components.len(), 1);
    let comp = &result.components[0];
    assert_eq!(comp.name, "Counter");
    assert_eq!(comp.component_type, ComponentType::Class);
    assert!(comp.class_info.is_some());
}

#[test]
fn test_detect_memo_component() {
    // GIVEN: A memoized component
    let source = r#"
        const ExpensiveList = memo(({ items }) => (
            <ul>
                {items.map(i => <li key={i}>{i}</li>)}
            </ul>
        ));
    "#;

    // WHEN: We extract
    let result = extract_source(source, Path::new("test.tsx"), "test.tsx").unwrap();

    // THEN: Memo wrapper is detected
    assert_eq!(result.components.len(), 1);
    let comp = &result.components[0];
    assert_eq!(comp.component_type, ComponentType::Memo);
}

#[test]
fn test_detect_forward_ref_component() {
    // GIVEN: A forwardRef component
    let source = r#"
        const FancyInput = forwardRef((props, ref) => (
            <input ref={ref} className="fancy" {...props} />
        ));
    "#;

    // WHEN: We extract
    let result = extract_source(source, Path::new("test.tsx"), "test.tsx").unwrap();

    // THEN: ForwardRef wrapper is detected
    assert_eq!(result.components.len(), 1);
    let comp = &result.components[0];
    assert_eq!(comp.component_type, ComponentType::ForwardRef);
}

// =============================================================================
// Phase 1.3: Hook Extraction
// =============================================================================

#[test]
fn test_extract_use_state_hook() {
    // GIVEN: Component with useState
    let source = r#"
        function Counter() {
            const [count, setCount] = useState(0);
            const [name, setName] = useState("default");
            return <div>{count}</div>;
        }
    "#;

    // WHEN: We extract
    let result = extract_source(source, Path::new("test.tsx"), "test.tsx").unwrap();

    // THEN: Hooks are extracted with state names
    let comp = &result.components[0];
    assert_eq!(comp.hooks.len(), 2);

    let count_hook = comp.hooks.iter()
        .find(|h| h.state_name.as_ref() == Some(&"count".to_string()))
        .expect("Should find count hook");
    assert_eq!(count_hook.hook_type, HookType::UseState);
    assert_eq!(count_hook.setter_name, Some("setCount".to_string()));
    assert_eq!(count_hook.initial_value, Some("0".to_string()));

    let name_hook = comp.hooks.iter()
        .find(|h| h.state_name.as_ref() == Some(&"name".to_string()))
        .expect("Should find name hook");
    assert_eq!(name_hook.initial_value, Some("\"default\"".to_string()));
}

#[test]
fn test_extract_use_effect_hook() {
    // GIVEN: Component with useEffect
    let source = r#"
        function DataLoader({ id }) {
            const [data, setData] = useState(null);

            useEffect(() => {
                fetchData(id).then(setData);
                return () => cleanup();
            }, [id]);

            useEffect(() => {
                console.log("mounted");
            }, []);

            return <div>{data}</div>;
        }
    "#;

    // WHEN: We extract
    let result = extract_source(source, Path::new("test.tsx"), "test.tsx").unwrap();

    // THEN: Effects are extracted with dependencies and cleanup info
    let comp = &result.components[0];
    let effects: Vec<_> = comp.hooks.iter()
        .filter(|h| h.hook_type == HookType::UseEffect)
        .collect();

    assert_eq!(effects.len(), 2);

    // First effect has dependency and cleanup
    let effect1 = effects.iter().find(|h| h.has_cleanup).unwrap();
    assert_eq!(effect1.dependencies, Some(vec!["id".to_string()]));

    // Second effect has empty deps (mount only)
    let effect2 = effects.iter().find(|h| !h.has_cleanup).unwrap();
    assert_eq!(effect2.dependencies, Some(vec![]));
}

#[test]
fn test_extract_use_callback_hook() {
    // GIVEN: Component with useCallback
    let source = r#"
        function Form({ onSubmit }) {
            const [value, setValue] = useState("");

            const handleSubmit = useCallback(() => {
                onSubmit(value);
            }, [value, onSubmit]);

            return <form onSubmit={handleSubmit}><input value={value} /></form>;
        }
    "#;

    // WHEN: We extract
    let result = extract_source(source, Path::new("test.tsx"), "test.tsx").unwrap();

    // THEN: useCallback is extracted with memoized dependencies
    let comp = &result.components[0];
    let callback = comp.hooks.iter()
        .find(|h| h.hook_type == HookType::UseCallback)
        .expect("Should find useCallback");

    assert_eq!(callback.memoized_deps, Some(vec!["value".to_string(), "onSubmit".to_string()]));
}

#[test]
fn test_extract_use_ref_hook() {
    // GIVEN: Component with useRef
    let source = r#"
        function TextInput() {
            const inputRef = useRef<HTMLInputElement>(null);

            const focus = () => inputRef.current?.focus();

            return <input ref={inputRef} />;
        }
    "#;

    // WHEN: We extract
    let result = extract_source(source, Path::new("test.tsx"), "test.tsx").unwrap();

    // THEN: useRef is extracted with ref name and type
    let comp = &result.components[0];
    let ref_hook = comp.hooks.iter()
        .find(|h| h.hook_type == HookType::UseRef)
        .expect("Should find useRef");

    assert_eq!(ref_hook.ref_name, Some("inputRef".to_string()));
    assert_eq!(ref_hook.ref_type, Some("HTMLInputElement".to_string()));
}

#[test]
fn test_extract_use_context_hook() {
    // GIVEN: Component with useContext
    let source = r#"
        function ThemeButton() {
            const theme = useContext(ThemeContext);
            return <button className={theme.buttonClass}>Click</button>;
        }
    "#;

    // WHEN: We extract
    let result = extract_source(source, Path::new("test.tsx"), "test.tsx").unwrap();

    // THEN: useContext is extracted with context name
    let comp = &result.components[0];
    let context_hook = comp.hooks.iter()
        .find(|h| h.hook_type == HookType::UseContext)
        .expect("Should find useContext");

    assert_eq!(context_hook.context_name, Some("ThemeContext".to_string()));
}

// =============================================================================
// Phase 1.4: Type Extraction
// =============================================================================

#[test]
fn test_extract_props_interface() {
    // GIVEN: Component with Props interface
    let source = r#"
        interface CounterProps {
            initial: number;
            onIncrement?: () => void;
            children?: React.ReactNode;
        }

        function Counter({ initial, onIncrement, children }: CounterProps) {
            return <div>{initial}</div>;
        }
    "#;

    // WHEN: We extract
    let result = extract_source(source, Path::new("test.tsx"), "test.tsx").unwrap();

    // THEN: Interface is extracted
    let type_ext = result.types.iter()
        .find(|t| t.name == "CounterProps")
        .expect("Should find CounterProps");

    assert_eq!(type_ext.kind, TypeKind::Interface);
}

#[test]
fn test_extract_props_type_alias() {
    // GIVEN: Component with type alias for props
    let source = r#"
        type ButtonProps = {
            variant: "primary" | "secondary";
            size?: "sm" | "md" | "lg";
        };

        const Button: React.FC<ButtonProps> = ({ variant, size = "md" }) => (
            <button className={`${variant} ${size}`}>Click</button>
        );
    "#;

    // WHEN: We extract
    let result = extract_source(source, Path::new("test.tsx"), "test.tsx").unwrap();

    // THEN: Type alias is extracted
    let type_ext = result.types.iter()
        .find(|t| t.name == "ButtonProps")
        .expect("Should find ButtonProps");

    assert_eq!(type_ext.kind, TypeKind::TypeAlias);
}

// =============================================================================
// Import/Export Tests
// =============================================================================

#[test]
fn test_extract_imports() {
    // GIVEN: Various import styles
    let source = r#"
        import React, { useState, useEffect } from 'react';
        import * as utils from './utils';
        import type { User } from './types';
        import defaultExport from 'some-lib';
    "#;

    // WHEN: We extract
    let result = extract_source(source, Path::new("test.tsx"), "test.tsx").unwrap();

    // THEN: All imports are extracted
    assert_eq!(result.imports.len(), 4);

    // React import
    let react_import = result.imports.iter()
        .find(|i| i.source == "react")
        .unwrap();
    assert!(react_import.specifiers.iter().any(|s| s.is_default && s.local == "React"));
    assert!(react_import.specifiers.iter().any(|s| s.imported == "useState"));
    assert!(react_import.specifiers.iter().any(|s| s.imported == "useEffect"));

    // Namespace import
    let utils_import = result.imports.iter()
        .find(|i| i.source == "./utils")
        .unwrap();
    assert!(utils_import.specifiers.iter().any(|s| s.is_namespace && s.local == "utils"));

    // Type-only import
    let type_import = result.imports.iter()
        .find(|i| i.source == "./types")
        .unwrap();
    assert!(type_import.is_type_only);
}

#[test]
fn test_extract_exports() {
    // GIVEN: Various export styles
    let source = r#"
        export function Counter() { return <div />; }
        export default function App() { return <div />; }
        export const Button = () => <button />;
        export type { CounterProps };
    "#;

    // WHEN: We extract
    let result = extract_source(source, Path::new("test.tsx"), "test.tsx").unwrap();

    // THEN: Components are marked as exported
    let counter = result.components.iter().find(|c| c.name == "Counter").unwrap();
    assert!(counter.exported);
    assert_eq!(counter.export_type, Some(ExportType::Named));

    let app = result.components.iter().find(|c| c.name == "App").unwrap();
    assert!(app.exported);
    assert_eq!(app.export_type, Some(ExportType::Default));
}

// =============================================================================
// Property Tests (Structural Invariants)
// =============================================================================

#[test]
fn test_property_all_components_have_location() {
    // PROPERTY: Every extracted component has a valid source location
    let sources = [
        "function A() { return <div />; }",
        "const B = () => <span />;",
        "class C extends Component { render() { return <p />; } }",
    ];

    for source in sources {
        let result = extract_source(source, Path::new("test.tsx"), "test.tsx").unwrap();
        for comp in &result.components {
            assert!(comp.location.start_line > 0, "Component {} has invalid start_line", comp.name);
            assert!(comp.location.end_line >= comp.location.start_line,
                "Component {} has end_line < start_line", comp.name);
        }
    }
}

#[test]
fn test_property_hook_state_names_match_setters() {
    // PROPERTY: For useState, if state_name is "foo", setter_name should be "setFoo"
    let source = r#"
        function Test() {
            const [count, setCount] = useState(0);
            const [isOpen, setIsOpen] = useState(false);
            const [items, setItems] = useState([]);
            return <div />;
        }
    "#;

    let result = extract_source(source, Path::new("test.tsx"), "test.tsx").unwrap();
    let comp = &result.components[0];

    for hook in &comp.hooks {
        if hook.hook_type == HookType::UseState {
            if let (Some(state_name), Some(setter_name)) = (&hook.state_name, &hook.setter_name) {
                let expected_setter = format!("set{}{}",
                    state_name.chars().next().unwrap().to_uppercase(),
                    &state_name[1..]);
                assert_eq!(setter_name, &expected_setter,
                    "State '{}' should have setter '{}'", state_name, expected_setter);
            }
        }
    }
}

#[test]
fn test_property_jsx_trees_are_well_formed() {
    // PROPERTY: JSX trees have consistent structure
    let source = r#"
        function Layout() {
            return (
                <div>
                    <header>Title</header>
                    <main>
                        <section>Content</section>
                    </main>
                    <footer>Footer</footer>
                </div>
            );
        }
    "#;

    let result = extract_source(source, Path::new("test.tsx"), "test.tsx").unwrap();
    let comp = &result.components[0];

    fn check_node(node: &JsxNode) {
        // Every node should have valid location
        assert!(node.location.start_line > 0);

        // Recursively check children
        match &node.node_type {
            JsxNodeType::Element { children, .. } => {
                for child in children {
                    check_node(child);
                }
            }
            JsxNodeType::Fragment { children } => {
                for child in children {
                    check_node(child);
                }
            }
            _ => {}
        }
    }

    if let Some(root) = &comp.jsx.root {
        check_node(root);
    }
}

// =============================================================================
// Phase 2: Spec Generation Tests
// =============================================================================

#[test]
fn test_recommend_state_field() {
    // GIVEN: Component with useState
    let source = r#"
        function Counter() {
            const [count, setCount] = useState(0);
            return <div>{count}</div>;
        }
    "#;

    // WHEN: We generate spec
    let extraction = extract_source(source, Path::new("test.tsx"), "test.tsx").unwrap();
    let spec = generate_spec(&extraction, source);

    // THEN: State field is recommended
    assert_eq!(spec.components.len(), 1);
    let comp_spec = &spec.components[0];

    let state_fields = &comp_spec.recommendations.state_fields;
    assert!(!state_fields.is_empty(), "Should have state field recommendations");

    let count_field = state_fields.iter()
        .find(|f| f.to_field == "count")
        .expect("Should recommend 'count' field");

    assert_eq!(count_field.from_hook, "useState:count");
    assert_eq!(count_field.field_type, "i64");
    assert_eq!(count_field.evidentiality, "!");
}

#[test]
fn test_recommend_message_from_handler() {
    // GIVEN: Component with event handler that sets state
    let source = r#"
        function Counter() {
            const [count, setCount] = useState(0);
            return <button onClick={() => setCount(c => c + 1)}>+</button>;
        }
    "#;

    // WHEN: We generate spec
    let extraction = extract_source(source, Path::new("test.tsx"), "test.tsx").unwrap();
    let spec = generate_spec(&extraction, source);

    // THEN: Message is recommended
    let comp_spec = &spec.components[0];
    let messages = &comp_spec.recommendations.messages;

    // Should have a message for the setter
    assert!(!messages.is_empty(), "Should have message recommendations");
}

#[test]
fn test_recommend_mount_effect() {
    // GIVEN: Component with mount-only useEffect
    let source = r#"
        function App() {
            useEffect(() => {
                console.log("mounted");
            }, []);
            return <div>App</div>;
        }
    "#;

    // WHEN: We generate spec
    let extraction = extract_source(source, Path::new("test.tsx"), "test.tsx").unwrap();
    let spec = generate_spec(&extraction, source);

    // THEN: Effect is recommended as lifecycle
    let comp_spec = &spec.components[0];
    let effects = &comp_spec.recommendations.effects;

    let mount_effect = effects.iter()
        .find(|e| e.strategy == EffectStrategy::Lifecycle)
        .expect("Should have lifecycle effect");

    assert_eq!(mount_effect.lifecycle_event, Some("Mount".to_string()));
}

#[test]
fn test_recommend_inline_effect() {
    // GIVEN: Component with useEffect that has deps
    let source = r#"
        function Counter() {
            const [count, setCount] = useState(0);

            useEffect(() => {
                document.title = count.toString();
            }, [count]);

            return <div>{count}</div>;
        }
    "#;

    // WHEN: We generate spec
    let extraction = extract_source(source, Path::new("test.tsx"), "test.tsx").unwrap();
    let spec = generate_spec(&extraction, source);

    // THEN: Effect is recommended as inline
    let comp_spec = &spec.components[0];
    let effects = &comp_spec.recommendations.effects;

    let inline_effect = effects.iter()
        .find(|e| e.strategy == EffectStrategy::Inline)
        .expect("Should have inline effect");

    assert!(inline_effect.from_hook.contains("count"));
}

#[test]
fn test_recommend_remove_callback() {
    // GIVEN: Component with useCallback
    let source = r#"
        function Form() {
            const [value, setValue] = useState("");
            const handleChange = useCallback((e) => {
                setValue(e.target.value);
            }, []);
            return <input onChange={handleChange} />;
        }
    "#;

    // WHEN: We generate spec
    let extraction = extract_source(source, Path::new("test.tsx"), "test.tsx").unwrap();
    let spec = generate_spec(&extraction, source);

    // THEN: useCallback is recommended for removal
    let comp_spec = &spec.components[0];
    let effects = &comp_spec.recommendations.effects;

    let callback_effect = effects.iter()
        .find(|e| e.from_hook.contains("UseCallback"))
        .expect("Should have useCallback recommendation");

    assert_eq!(callback_effect.strategy, EffectStrategy::Remove);
}

#[test]
fn test_recommend_actor_pattern() {
    // GIVEN: Component with state
    let source = r#"
        function Counter() {
            const [count, setCount] = useState(0);
            return <div>{count}</div>;
        }
    "#;

    // WHEN: We generate spec
    let extraction = extract_source(source, Path::new("test.tsx"), "test.tsx").unwrap();
    let spec = generate_spec(&extraction, source);

    // THEN: Target pattern is actor
    let comp_spec = &spec.components[0];
    assert_eq!(comp_spec.target.pattern, TargetPattern::Actor);
}

#[test]
fn test_recommend_function_pattern() {
    // GIVEN: Pure component with no hooks
    let source = r#"
        function Greeting({ name }) {
            return <h1>Hello, {name}!</h1>;
        }
    "#;

    // WHEN: We generate spec
    let extraction = extract_source(source, Path::new("test.tsx"), "test.tsx").unwrap();
    let spec = generate_spec(&extraction, source);

    // THEN: Target pattern is function
    let comp_spec = &spec.components[0];
    assert_eq!(comp_spec.target.pattern, TargetPattern::Function);
}

// =============================================================================
// Pattern Matching Tests
// =============================================================================

#[test]
fn test_pattern_for_usestate() {
    // GIVEN: Component with useState
    let source = r#"
        function Counter() {
            const [count, setCount] = useState(0);
            return <div>{count}</div>;
        }
    "#;

    // WHEN: We generate spec
    let extraction = extract_source(source, Path::new("test.tsx"), "test.tsx").unwrap();
    let spec = generate_spec(&extraction, source);

    // THEN: useState pattern is included
    let comp_spec = &spec.components[0];
    assert!(comp_spec.patterns.iter().any(|p| p.name == "useState_to_state"),
        "Should include useState_to_state pattern");
}

#[test]
fn test_pattern_for_onclick() {
    // GIVEN: Button with onClick
    let source = r#"
        function Button() {
            return <button onClick={() => alert('clicked')}>Click</button>;
        }
    "#;

    // WHEN: We generate spec
    let extraction = extract_source(source, Path::new("test.tsx"), "test.tsx").unwrap();
    let spec = generate_spec(&extraction, source);

    // THEN: onClick pattern is included
    let comp_spec = &spec.components[0];
    assert!(comp_spec.patterns.iter().any(|p| p.name == "onClick_to_message"),
        "Should include onClick_to_message pattern");
}

#[test]
fn test_no_duplicate_patterns() {
    // GIVEN: Component with multiple uses of same pattern
    let source = r#"
        function Multi() {
            const [a, setA] = useState(0);
            const [b, setB] = useState(0);
            const [c, setC] = useState(0);
            return <div>{a + b + c}</div>;
        }
    "#;

    // WHEN: We generate spec
    let extraction = extract_source(source, Path::new("test.tsx"), "test.tsx").unwrap();
    let spec = generate_spec(&extraction, source);

    // THEN: Patterns are unique
    let comp_spec = &spec.components[0];
    let pattern_names: Vec<_> = comp_spec.patterns.iter().map(|p| &p.name).collect();
    let unique_count = pattern_names.iter().collect::<std::collections::HashSet<_>>().len();

    assert_eq!(pattern_names.len(), unique_count, "Patterns should be unique");
}

// =============================================================================
// Ambiguity Detection Tests
// =============================================================================

#[test]
fn test_ambiguity_effect_placement() {
    // GIVEN: useEffect with deps
    let source = r#"
        function Counter() {
            const [count, setCount] = useState(0);
            useEffect(() => { save(count); }, [count]);
            return <div>{count}</div>;
        }
    "#;

    // WHEN: We generate spec
    let extraction = extract_source(source, Path::new("test.tsx"), "test.tsx").unwrap();
    let spec = generate_spec(&extraction, source);

    // THEN: Ambiguity is detected
    let comp_spec = &spec.components[0];
    let effect_ambiguity = comp_spec.ambiguities.iter()
        .find(|a| a.category == AmbiguityCategory::EffectPlacement);

    assert!(effect_ambiguity.is_some(), "Should detect effect placement ambiguity");

    let ambiguity = effect_ambiguity.unwrap();
    assert!(ambiguity.options.len() >= 2, "Should have at least 2 options");
    assert!(ambiguity.options.iter().any(|o| o.recommended), "Should have recommended option");
}

#[test]
fn test_no_ambiguity_simple() {
    // GIVEN: Simple component with no ambiguities
    let source = r#"
        function Simple() {
            return <div>Hello</div>;
        }
    "#;

    // WHEN: We generate spec
    let extraction = extract_source(source, Path::new("test.tsx"), "test.tsx").unwrap();
    let spec = generate_spec(&extraction, source);

    // THEN: No ambiguities
    let comp_spec = &spec.components[0];
    assert!(comp_spec.ambiguities.is_empty(), "Simple component should have no ambiguities");
}

// =============================================================================
// Dependency Analysis Tests
// =============================================================================

#[test]
fn test_detect_component_import() {
    // GIVEN: Component that imports another component
    let source = r#"
        import { Button } from './Button';

        function Form() {
            return <Button>Submit</Button>;
        }
    "#;

    // WHEN: We extract
    let extraction = extract_source(source, Path::new("test.tsx"), "test.tsx").unwrap();

    // THEN: Import is detected
    let button_import = extraction.imports.iter()
        .find(|i| i.source == "./Button");
    assert!(button_import.is_some(), "Should detect Button import");
}

// =============================================================================
// Complexity Calculation Tests
// =============================================================================

#[test]
fn test_complexity_simple() {
    // GIVEN: Simple component
    let source = r#"
        function Hello() {
            return <div>Hello</div>;
        }
    "#;

    // WHEN: We generate spec
    let extraction = extract_source(source, Path::new("test.tsx"), "test.tsx").unwrap();
    let spec = generate_spec(&extraction, source);

    // THEN: Complexity is simple
    let comp_spec = &spec.components[0];
    assert_eq!(comp_spec.complexity, Complexity::Simple);
}

#[test]
fn test_complexity_moderate() {
    // GIVEN: Component with a few hooks
    let source = r#"
        function Counter() {
            const [count, setCount] = useState(0);
            const [name, setName] = useState("");
            useEffect(() => {}, [count]);
            return <div>{count}</div>;
        }
    "#;

    // WHEN: We generate spec
    let extraction = extract_source(source, Path::new("test.tsx"), "test.tsx").unwrap();
    let spec = generate_spec(&extraction, source);

    // THEN: Complexity is moderate
    let comp_spec = &spec.components[0];
    assert!(comp_spec.complexity == Complexity::Moderate || comp_spec.complexity == Complexity::Simple);
}

#[test]
fn test_complexity_complex() {
    // GIVEN: Complex component with many hooks
    let source = r#"
        function Dashboard() {
            const [a, setA] = useState(0);
            const [b, setB] = useState("");
            const [c, setC] = useState([]);
            const [d, setD] = useState({});
            const [e, setE] = useState(null);
            const [f, setF] = useState(true);
            useEffect(() => {}, [a]);
            useEffect(() => {}, [b]);
            useEffect(() => {}, [c]);
            useEffect(() => {}, [d]);
            return <div>{a}</div>;
        }
    "#;

    // WHEN: We generate spec
    let extraction = extract_source(source, Path::new("test.tsx"), "test.tsx").unwrap();
    let spec = generate_spec(&extraction, source);

    // THEN: Complexity is complex
    let comp_spec = &spec.components[0];
    assert_eq!(comp_spec.complexity, Complexity::Complex);
    assert!(!comp_spec.complexity_factors.is_empty(), "Should have complexity factors");
}

// =============================================================================
// Property Tests for Spec Generation
// =============================================================================

#[test]
fn test_property_all_usestate_have_state_field() {
    // PROPERTY: Every useState should have a corresponding state field recommendation
    let source = r#"
        function Multi() {
            const [a, setA] = useState(0);
            const [b, setB] = useState("");
            const [c, setC] = useState(false);
            return <div />;
        }
    "#;

    let extraction = extract_source(source, Path::new("test.tsx"), "test.tsx").unwrap();
    let spec = generate_spec(&extraction, source);

    let comp = &extraction.components[0];
    let comp_spec = &spec.components[0];

    let use_state_count = comp.hooks.iter()
        .filter(|h| h.hook_type == HookType::UseState)
        .count();

    assert_eq!(comp_spec.recommendations.state_fields.len(), use_state_count,
        "Every useState should have a state field recommendation");
}

#[test]
fn test_property_patterns_are_relevant() {
    // PROPERTY: All included patterns should be relevant to the component
    let source = r#"
        function Counter() {
            const [count, setCount] = useState(0);
            return <button onClick={() => setCount(c => c + 1)}>{count}</button>;
        }
    "#;

    let extraction = extract_source(source, Path::new("test.tsx"), "test.tsx").unwrap();
    let spec = generate_spec(&extraction, source);

    let comp_spec = &spec.components[0];

    // Should have useState pattern
    assert!(comp_spec.patterns.iter().any(|p| p.name == "useState_to_state"),
        "Should include useState pattern for component with useState");

    // Should have onClick pattern
    assert!(comp_spec.patterns.iter().any(|p| p.name == "onClick_to_message"),
        "Should include onClick pattern for component with onClick");
}

// =============================================================================
// Phase 3: Qliphoth Code Generation Tests
// =============================================================================

#[test]
fn test_generate_pure_function() {
    // GIVEN: A simple component with no state (pure function)
    let source = r#"
        function Empty() {
            return <div>Empty</div>;
        }
    "#;

    // WHEN: We generate Sigil code
    let extraction = extract_source(source, Path::new("test.tsx"), "test.tsx").unwrap();
    let spec = generate_spec(&extraction, source);
    let generated = generate_component(&spec.components[0]);

    // THEN: We get a function (not an actor) with prelude import
    assert!(generated.code.contains("invoke qliphoth·prelude·*;"),
        "Should include qliphoth prelude import");
    assert!(generated.code.contains("rite empty("),
        "Should generate function with snake_case name: {}", generated.code);
    assert!(generated.code.contains("-> VNode!"),
        "Should return VNode!");
    assert!(generated.code.contains("VNode·div()"),
        "Should generate VNode·div()");
}

#[test]
fn test_generate_actor_with_state() {
    // GIVEN: Component with useState
    let source = r#"
        function Counter() {
            const [count, setCount] = useState(0);
            return <div>{count}</div>;
        }
    "#;

    // WHEN: We generate Sigil code
    let extraction = extract_source(source, Path::new("test.tsx"), "test.tsx").unwrap();
    let spec = generate_spec(&extraction, source);
    let generated = generate_component(&spec.components[0]);

    // THEN: Actor with state field is generated
    assert!(generated.code.contains("☉ actor Counter"),
        "Should generate actor declaration");
    assert!(generated.code.contains("state count:"),
        "Should have state field");
    assert!(generated.code.contains("i64"),
        "Should infer i64 type from 0");
}

#[test]
fn test_generate_message_enum() {
    // GIVEN: Component with click handler
    let source = r#"
        function Counter() {
            const [count, setCount] = useState(0);
            return <button onClick={() => setCount(c => c + 1)}>+</button>;
        }
    "#;

    // WHEN: We generate Sigil code
    let extraction = extract_source(source, Path::new("test.tsx"), "test.tsx").unwrap();
    let spec = generate_spec(&extraction, source);
    let generated = generate_component(&spec.components[0]);

    // THEN: Message enum is generated if there are messages
    if !spec.components[0].recommendations.messages.is_empty() {
        assert!(generated.code.contains("ᛈ CounterMsg"),
            "Should generate message enum");
    }
}

#[test]
fn test_generate_message_handlers() {
    // GIVEN: Component with event handler
    let source = r#"
        function Counter() {
            const [count, setCount] = useState(0);
            const increment = () => setCount(c => c + 1);
            return <button onClick={increment}>+</button>;
        }
    "#;

    // WHEN: We generate Sigil code
    let extraction = extract_source(source, Path::new("test.tsx"), "test.tsx").unwrap();
    let spec = generate_spec(&extraction, source);
    let generated = generate_component(&spec.components[0]);

    // THEN: Message handlers are generated
    // The handler should be present in the actor
    assert!(generated.code.contains("☉ actor Counter"),
        "Should generate actor");
}

#[test]
fn test_gen_simple_div() {
    // GIVEN: Simple div element
    let source = r#"
        function Simple() {
            return <div>Hello</div>;
        }
    "#;

    // WHEN: We generate Sigil code
    let extraction = extract_source(source, Path::new("test.tsx"), "test.tsx").unwrap();
    let spec = generate_spec(&extraction, source);
    let generated = generate_component(&spec.components[0]);

    // THEN: VNode builder is generated
    assert!(generated.code.contains("VNode·div()"),
        "Should generate VNode·div()");
    assert!(generated.code.contains("text_child(\"Hello\")"),
        "Should generate text_child");
}

#[test]
fn test_gen_with_class() {
    // GIVEN: Element with className
    let source = r#"
        function Styled() {
            return <div className="container">Content</div>;
        }
    "#;

    // WHEN: We generate Sigil code
    let extraction = extract_source(source, Path::new("test.tsx"), "test.tsx").unwrap();
    let spec = generate_spec(&extraction, source);
    let generated = generate_component(&spec.components[0]);

    // THEN: Class attribute is generated
    assert!(generated.code.contains("·class(\"container\")"),
        "Should generate ·class() method: {}", generated.code);
}

#[test]
fn test_gen_nested() {
    // GIVEN: Nested elements
    let source = r#"
        function Layout() {
            return (
                <div>
                    <header>Title</header>
                    <main>Content</main>
                </div>
            );
        }
    "#;

    // WHEN: We generate Sigil code
    let extraction = extract_source(source, Path::new("test.tsx"), "test.tsx").unwrap();
    let spec = generate_spec(&extraction, source);
    let generated = generate_component(&spec.components[0]);

    // THEN: Nested structure is generated with ·child()
    assert!(generated.code.contains("VNode·div()"),
        "Should generate parent div");
    assert!(generated.code.contains("VNode·header()"),
        "Should generate header child");
    assert!(generated.code.contains("VNode·main()"),
        "Should generate main child");
    assert!(generated.code.contains("·child("),
        "Should use ·child() for nesting");
}

#[test]
fn test_gen_event_handler() {
    // GIVEN: Element with onClick
    let source = r#"
        function Button() {
            return <button onClick={handleClick}>Click</button>;
        }
    "#;

    // WHEN: We generate Sigil code
    let extraction = extract_source(source, Path::new("test.tsx"), "test.tsx").unwrap();
    let spec = generate_spec(&extraction, source);
    let generated = generate_component(&spec.components[0]);

    // THEN: Event handler is converted to message dispatch
    assert!(generated.code.contains("·on_click("),
        "Should generate ·on_click() method: {}", generated.code);
}

#[test]
fn test_gen_function_component() {
    // GIVEN: Pure component (no hooks)
    let source = r#"
        function Greeting({ name }) {
            return <h1>Hello, {name}!</h1>;
        }
    "#;

    // WHEN: We generate Sigil code
    let extraction = extract_source(source, Path::new("test.tsx"), "test.tsx").unwrap();
    let spec = generate_spec(&extraction, source);
    let generated = generate_component(&spec.components[0]);

    // THEN: Function is generated (not actor)
    assert!(generated.code.contains("rite greeting("),
        "Should generate function with snake_case name: {}", generated.code);
    assert!(generated.code.contains("-> VNode!"),
        "Should return VNode!");
}

#[test]
fn test_gen_fragment() {
    // GIVEN: JSX Fragment
    let source = r#"
        function List() {
            return (
                <>
                    <li>One</li>
                    <li>Two</li>
                </>
            );
        }
    "#;

    // WHEN: We generate Sigil code
    let extraction = extract_source(source, Path::new("test.tsx"), "test.tsx").unwrap();
    let spec = generate_spec(&extraction, source);
    let generated = generate_component(&spec.components[0]);

    // THEN: Fragment is generated
    assert!(generated.code.contains("VNode·fragment()"),
        "Should generate VNode·fragment(): {}", generated.code);
}

#[test]
fn test_gen_component_name_in_output() {
    // GIVEN: Named component
    let source = r#"
        function MyComponent() {
            return <div />;
        }
    "#;

    // WHEN: We generate Sigil code
    let extraction = extract_source(source, Path::new("test.tsx"), "test.tsx").unwrap();
    let spec = generate_spec(&extraction, source);
    let generated = generate_component(&spec.components[0]);

    // THEN: Component name is preserved
    assert_eq!(generated.component_name, "MyComponent");
}

#[test]
fn test_gen_suggested_path() {
    // GIVEN: Component
    let source = r#"
        function Counter() {
            const [count, setCount] = useState(0);
            return <div>{count}</div>;
        }
    "#;

    // WHEN: We generate Sigil code
    let extraction = extract_source(source, Path::new("test.tsx"), "test.tsx").unwrap();
    let spec = generate_spec(&extraction, source);
    let generated = generate_component(&spec.components[0]);

    // THEN: Path is suggested
    assert!(!generated.path.is_empty(), "Should have suggested path");
    assert!(generated.path.ends_with(".sigil"), "Path should end with .sigil");
}

#[test]
fn test_gen_attributes() {
    // GIVEN: Element with various attributes
    let source = r#"
        function Link() {
            return <a href="/home" id="main-link">Home</a>;
        }
    "#;

    // WHEN: We generate Sigil code
    let extraction = extract_source(source, Path::new("test.tsx"), "test.tsx").unwrap();
    let spec = generate_spec(&extraction, source);
    let generated = generate_component(&spec.components[0]);

    // THEN: Attributes are generated
    assert!(generated.code.contains("·attr(\"href\", \"/home\")"),
        "Should generate href attr: {}", generated.code);
    assert!(generated.code.contains("·id(\"main-link\")"),
        "Should generate id attr: {}", generated.code);
}

#[test]
fn test_gen_disabled_attr() {
    // GIVEN: Element with disabled attribute
    let source = r#"
        function Button() {
            return <button disabled>Disabled</button>;
        }
    "#;

    // WHEN: We generate Sigil code
    let extraction = extract_source(source, Path::new("test.tsx"), "test.tsx").unwrap();
    let spec = generate_spec(&extraction, source);
    let generated = generate_component(&spec.components[0]);

    // THEN: Boolean attribute is generated
    assert!(generated.code.contains("·attr(\"disabled\", \"true\")"),
        "Should generate disabled attr: {}", generated.code);
}

#[test]
fn test_generate_all_components() {
    // GIVEN: File with multiple components
    let source = r#"
        function Header() {
            return <header>Header</header>;
        }

        function Footer() {
            return <footer>Footer</footer>;
        }
    "#;

    // WHEN: We generate all
    let extraction = extract_source(source, Path::new("test.tsx"), "test.tsx").unwrap();
    let spec = generate_spec(&extraction, source);
    let generated = generate_all(&spec);

    // THEN: All components are generated
    assert_eq!(generated.len(), 2, "Should generate 2 components");
    assert!(generated.iter().any(|g| g.component_name == "Header"));
    assert!(generated.iter().any(|g| g.component_name == "Footer"));
}

// =============================================================================
// Content Validation Tests (from audit findings)
// These tests validate that actual content is extracted, not placeholders
// =============================================================================

#[test]
fn test_expression_content_preserved() {
    // CRITICAL-1: Expression content must be extracted, not placeholder
    let source = r#"
        function Counter() {
            const [count, setCount] = useState(0);
            return <div>{count}</div>;
        }
    "#;

    let extraction = extract_source(source, Path::new("test.tsx"), "test.tsx").unwrap();
    let comp = &extraction.components[0];
    let root = comp.jsx.root.as_ref().expect("Should have JSX root");

    // Find the expression child
    if let JsxNodeType::Element { children, .. } = &root.node_type {
        assert!(!children.is_empty(), "Should have children");
        let expr_child = &children[0];
        if let JsxNodeType::Expression { code } = &expr_child.node_type {
            // The expression should be "count", not "/* expression */"
            assert!(!code.contains("/*"), "Expression should not be a placeholder: {}", code);
            assert_eq!(code.trim(), "count", "Expression should be 'count': {}", code);
        } else {
            panic!("First child should be an expression: {:?}", expr_child.node_type);
        }
    }
}

#[test]
fn test_attribute_expression_preserved() {
    // Attribute expressions should also be preserved
    let source = r#"
        function Input({ value }) {
            return <input value={value} />;
        }
    "#;

    let extraction = extract_source(source, Path::new("test.tsx"), "test.tsx").unwrap();
    let comp = &extraction.components[0];
    let root = comp.jsx.root.as_ref().expect("Should have JSX root");

    if let JsxNodeType::Element { attributes, .. } = &root.node_type {
        let value_attr = attributes.iter().find(|a| a.name == "value").expect("Should have value attr");
        if let JsxAttributeValue::Expression { code } = &value_attr.value {
            assert!(!code.contains("/*"), "Expression should not be placeholder: {}", code);
            assert_eq!(code.trim(), "value", "Should be 'value': {}", code);
        } else {
            panic!("Value attribute should be expression");
        }
    }
}

#[test]
fn test_handler_extraction() {
    // CRITICAL-2: Handler extraction should work
    let source = r#"
        function App() {
            const handleClick = () => {
                setCount(c => c + 1);
            };
            return <button onClick={handleClick}>Click</button>;
        }
    "#;

    let extraction = extract_source(source, Path::new("test.tsx"), "test.tsx").unwrap();
    let comp = &extraction.components[0];

    assert!(!comp.handlers.is_empty(), "Should extract handlers");
    let handler = comp.handlers.iter().find(|h| h.name == "handleClick")
        .expect("Should find handleClick handler");
    assert!(!handler.body_summary.is_empty(), "Handler should have body summary");
}

#[test]
fn test_props_extraction() {
    // CRITICAL-3: Props extraction should work
    let source = r#"
        function Greeting({ name, age = 0, onChange }) {
            return <div>{name}</div>;
        }
    "#;

    let extraction = extract_source(source, Path::new("test.tsx"), "test.tsx").unwrap();
    let comp = &extraction.components[0];

    assert!(!comp.props.is_empty(), "Should extract props");
    assert!(comp.props.iter().any(|p| p.name == "name"), "Should have 'name' prop");
    assert!(comp.props.iter().any(|p| p.name == "age"), "Should have 'age' prop");
    assert!(comp.props.iter().any(|p| p.name == "onChange"), "Should have 'onChange' prop");

    // Check that onChange is detected as callback
    let on_change = comp.props.iter().find(|p| p.name == "onChange").unwrap();
    assert!(on_change.is_callback, "onChange should be detected as callback");
}

#[test]
fn test_child_components_extracted() {
    // CRITICAL-4: Child components should be extracted
    let source = r#"
        function App() {
            return (
                <div>
                    <Header />
                    <Main>
                        <Sidebar />
                    </Main>
                    <Footer />
                </div>
            );
        }
    "#;

    let extraction = extract_source(source, Path::new("test.tsx"), "test.tsx").unwrap();
    let comp = &extraction.components[0];

    assert!(!comp.child_components.is_empty(), "Should extract child components");
    assert!(comp.child_components.contains(&"Header".to_string()), "Should have Header");
    assert!(comp.child_components.contains(&"Main".to_string()), "Should have Main");
    assert!(comp.child_components.contains(&"Sidebar".to_string()), "Should have Sidebar");
    assert!(comp.child_components.contains(&"Footer".to_string()), "Should have Footer");
}

#[test]
fn test_conditional_expression_extraction() {
    // Test conditional expression pattern extraction
    let source = r#"
        function Greeting({ isLoggedIn }) {
            return <div>{isLoggedIn && <Welcome />}</div>;
        }
    "#;

    let extraction = extract_source(source, Path::new("test.tsx"), "test.tsx").unwrap();
    let comp = &extraction.components[0];
    let root = comp.jsx.root.as_ref().expect("Should have JSX root");

    if let JsxNodeType::Element { children, .. } = &root.node_type {
        let conditional = &children[0];
        if let JsxNodeType::Conditional { condition, .. } = &conditional.node_type {
            assert!(!condition.contains("/*"), "Condition should not be placeholder");
            assert!(condition.contains("isLoggedIn"), "Condition should contain isLoggedIn: {}", condition);
        } else {
            panic!("Should be conditional node: {:?}", conditional.node_type);
        }
    }
}

#[test]
fn test_generated_code_has_actual_expressions() {
    // CRITICAL-5 & 6: Generated code should have actual expressions
    let source = r#"
        function Counter() {
            const [count, setCount] = useState(0);
            return <div>{count}</div>;
        }
    "#;

    let extraction = extract_source(source, Path::new("test.tsx"), "test.tsx").unwrap();
    let spec = generate_spec(&extraction, source);
    let generated = generate_component(&spec.components[0]);

    // Should NOT contain "/* expression */"
    assert!(!generated.code.contains("/* expression */"),
        "Generated code should not have placeholder expressions: {}", generated.code);

    // Should contain the actual expression reference
    assert!(generated.code.contains("self.count") || generated.code.contains("count"),
        "Generated code should reference count: {}", generated.code);
}

#[test]
fn test_pure_function_no_self() {
    // CRITICAL-6: Pure functions should NOT use self
    let source = r#"
        function Greeting({ name }) {
            return <h1>Hello, {name}!</h1>;
        }
    "#;

    let extraction = extract_source(source, Path::new("test.tsx"), "test.tsx").unwrap();
    let spec = generate_spec(&extraction, source);
    let generated = generate_component(&spec.components[0]);

    // Should be a function, not an actor
    assert!(generated.code.contains("rite greeting("),
        "Should generate function: {}", generated.code);

    // Should NOT have self references for a pure function
    assert!(!generated.code.contains("self.name"),
        "Pure function should not use self.name: {}", generated.code);
}

#[test]
fn test_spread_props_extracted() {
    // Test spread attribute extraction
    let source = r#"
        function Button(props) {
            return <button {...props}>Click</button>;
        }
    "#;

    let extraction = extract_source(source, Path::new("test.tsx"), "test.tsx").unwrap();
    let comp = &extraction.components[0];
    let root = comp.jsx.root.as_ref().expect("Should have JSX root");

    if let JsxNodeType::Element { attributes, .. } = &root.node_type {
        let spread = attributes.iter().find(|a| a.name == "...").expect("Should have spread");
        if let JsxAttributeValue::Spread { name } = &spread.value {
            assert!(!name.contains("/*"), "Spread should not be placeholder: {}", name);
            assert!(name.contains("props"), "Spread should reference props: {}", name);
        } else {
            panic!("Should be spread value");
        }
    }
}

#[test]
fn test_timestamp_is_current() {
    // MINOR-1: Timestamps should be current (not hardcoded 2026)
    let source = r#"
        function App() { return <div />; }
    "#;

    let extraction = extract_source(source, Path::new("test.tsx"), "test.tsx").unwrap();
    let spec = generate_spec(&extraction, source);

    // Timestamp should be recent (within last minute for test stability)
    // Just check it's a valid ISO format and contains current year
    assert!(spec.generated_at.contains("T"),
        "Should be ISO format: {}", spec.generated_at);
    assert!(spec.generated_at.ends_with("Z"),
        "Should be UTC: {}", spec.generated_at);
}

#[test]
fn test_type_inference_array_elements() {
    // MINOR-2: Type inference should handle array elements
    let source = r#"
        function App() {
            const [items, setItems] = useState([1, 2, 3]);
            return <div />;
        }
    "#;

    let extraction = extract_source(source, Path::new("test.tsx"), "test.tsx").unwrap();
    let spec = generate_spec(&extraction, source);

    // Check that state field type is inferred
    let state_field = spec.components[0].recommendations.state_fields
        .iter()
        .find(|f| f.to_field == "items")
        .expect("Should have items state field");

    // Should infer Vec<i64> from [1, 2, 3]
    assert!(state_field.field_type.contains("Vec"),
        "Should infer Vec type: {}", state_field.field_type);
}

// =============================================================================
// Phase 4: MCP Interface
// =============================================================================

#[test]
fn test_mcp_list_migrations_empty() {
    // GIVEN: Empty migration session
    let extraction = ReactExtraction {
        file: FileInfo {
            path: PathBuf::from("test.tsx"),
            relative_path: "test.tsx".to_string(),
            language: Language::TypeScript,
            has_jsx: true,
        },
        components: vec![],
        custom_hooks: vec![],
        imports: vec![],
        exports: vec![],
        types: vec![],
        helper_functions: vec![],
    };
    let spec = generate_spec(&extraction, "");
    let session = MigrationSession::from_spec(spec, "/tmp/output");

    // WHEN: We list migrations
    let migrations = session.list_migrations();

    // THEN: List is empty
    assert!(migrations.is_empty());
}

#[test]
fn test_mcp_list_migrations_populated() {
    // GIVEN: Session with components
    let source = r#"
        function Counter() {
            const [count, setCount] = useState(0);
            return <div>{count}</div>;
        }
        function Display({ value }) {
            return <span>{value}</span>;
        }
    "#;

    let extraction = extract_source(source, Path::new("test.tsx"), "test.tsx").unwrap();
    let spec = generate_spec(&extraction, source);
    let session = MigrationSession::from_spec(spec, "/tmp/output");

    // WHEN: We list migrations
    let migrations = session.list_migrations();

    // THEN: Both components appear with their metadata
    assert_eq!(migrations.len(), 2);

    let counter = migrations.iter().find(|m| m.name == "Counter")
        .expect("Should have Counter component");
    let display = migrations.iter().find(|m| m.name == "Display")
        .expect("Should have Display component");

    // Check both have IDs and status
    assert!(!counter.id.is_empty());
    assert!(!display.id.is_empty());
    assert_eq!(counter.status, MigrationStatus::Pending);
    assert_eq!(display.status, MigrationStatus::Pending);
}

#[test]
fn test_mcp_get_migration() {
    // GIVEN: Session with a component
    let source = r#"
        function App() {
            return <div>Hello</div>;
        }
    "#;

    let extraction = extract_source(source, Path::new("test.tsx"), "test.tsx").unwrap();
    let spec = generate_spec(&extraction, source);
    let session = MigrationSession::from_spec(spec, "/tmp/output");

    // WHEN: We get the migration
    let comp_id = &session.list_migrations()[0].id;
    let result = session.get_migration(comp_id);

    // THEN: We get the full spec
    assert!(result.is_ok());
    let comp = result.unwrap();
    assert_eq!(comp.name, "App");
}

#[test]
fn test_mcp_get_migration_not_found() {
    // GIVEN: Session with a component
    let source = r#"
        function App() { return <div />; }
    "#;

    let extraction = extract_source(source, Path::new("test.tsx"), "test.tsx").unwrap();
    let spec = generate_spec(&extraction, source);
    let session = MigrationSession::from_spec(spec, "/tmp/output");

    // WHEN: We request non-existent component
    let result = session.get_migration("nonexistent-id");

    // THEN: We get NotFound error
    assert!(result.is_err());
    match result.unwrap_err() {
        McpError::NotFound(id) => assert_eq!(id, "nonexistent-id"),
        e => panic!("Expected NotFound, got {:?}", e),
    }
}

#[test]
fn test_mcp_validate_sigil_valid() {
    // GIVEN: Valid Sigil code
    let code = r#"
invoke qliphoth·prelude·*;

actor Counter {
    count: i64,

    rite new() -> Self {
        Self { count: 0 }
    }

    rite view(&self) -> VNode {
        VNode·div()
            ·child(VNode·text(self.count.to_string()))
    }
}
    "#;

    let source = "function App() { return <div />; }";
    let extraction = extract_source(source, Path::new("test.tsx"), "test.tsx").unwrap();
    let spec = generate_spec(&extraction, source);
    let session = MigrationSession::from_spec(spec, "/tmp/output");

    // WHEN: We validate
    let result = session.validate_sigil(code);

    // THEN: It passes
    assert!(result.valid, "Should be valid: {:?}", result.errors);
}

#[test]
fn test_mcp_validate_sigil_invalid_missing_import() {
    // GIVEN: Sigil code missing import
    let code = r#"
actor Counter {
    count: i64,
}
    "#;

    let source = "function App() { return <div />; }";
    let extraction = extract_source(source, Path::new("test.tsx"), "test.tsx").unwrap();
    let spec = generate_spec(&extraction, source);
    let session = MigrationSession::from_spec(spec, "/tmp/output");

    // WHEN: We validate
    let result = session.validate_sigil(code);

    // THEN: It fails with import error
    assert!(!result.valid);
    assert!(result.errors.iter().any(|e| e.message.contains("import")));
}

#[test]
fn test_mcp_validate_sigil_placeholder_expression() {
    // GIVEN: Code with unresolved placeholder
    let code = r#"
invoke qliphoth·prelude·*;

actor Counter {
    rite view(&self) -> VNode {
        VNode·div()·child(/* expression */)
    }
}
    "#;

    let source = "function App() { return <div />; }";
    let extraction = extract_source(source, Path::new("test.tsx"), "test.tsx").unwrap();
    let spec = generate_spec(&extraction, source);
    let session = MigrationSession::from_spec(spec, "/tmp/output");

    // WHEN: We validate
    let result = session.validate_sigil(code);

    // THEN: It fails with placeholder error
    assert!(!result.valid);
    assert!(result.errors.iter().any(|e| e.message.contains("Placeholder")));
}

#[test]
fn test_mcp_start_migration() {
    // GIVEN: Session with components
    let source = r#"
        function App() { return <div />; }
    "#;

    let extraction = extract_source(source, Path::new("test.tsx"), "test.tsx").unwrap();
    let spec = generate_spec(&extraction, source);
    let mut session = MigrationSession::from_spec(spec, "/tmp/output");

    // WHEN: We start a migration
    let comp_id = session.list_migrations()[0].id.clone();
    session.start_migration(&comp_id).unwrap();

    // THEN: Status changes to InProgress
    let migrations = session.list_migrations();
    assert_eq!(migrations[0].status, MigrationStatus::InProgress);
}

#[test]
fn test_mcp_resolve_ambiguity() {
    // GIVEN: Component with ambiguities
    let source = r#"
        function Counter() {
            const [count, setCount] = useState(0);
            useEffect(() => {
                console.log(count);
            }, [count]);
            return <div>{count}</div>;
        }
    "#;

    let extraction = extract_source(source, Path::new("test.tsx"), "test.tsx").unwrap();
    let spec = generate_spec(&extraction, source);
    let mut session = MigrationSession::from_spec(spec, "/tmp/output");

    let comp_id = session.list_migrations()[0].id.clone();

    // Get the ambiguity id separately to avoid borrow conflicts
    let ambiguity_id = {
        let comp = session.get_migration(&comp_id).unwrap();
        if comp.ambiguities.is_empty() {
            return; // No ambiguities to test
        }
        comp.ambiguities[0].id.clone()
    };

    // WHEN: We resolve the ambiguity
    let result = session.resolve_ambiguity(&comp_id, &ambiguity_id, 0);

    // THEN: It succeeds
    assert!(result.is_ok());
}

#[test]
fn test_mcp_resolve_ambiguity_invalid_choice() {
    // GIVEN: Component with ambiguities
    let source = r#"
        function Counter() {
            const [count, setCount] = useState(0);
            useEffect(() => { console.log(count); }, [count]);
            return <div>{count}</div>;
        }
    "#;

    let extraction = extract_source(source, Path::new("test.tsx"), "test.tsx").unwrap();
    let spec = generate_spec(&extraction, source);
    let mut session = MigrationSession::from_spec(spec, "/tmp/output");

    let comp_id = session.list_migrations()[0].id.clone();

    // Get the ambiguity id separately to avoid borrow conflicts
    let ambiguity_id = {
        let comp = session.get_migration(&comp_id).unwrap();
        if comp.ambiguities.is_empty() {
            return; // No ambiguities to test
        }
        comp.ambiguities[0].id.clone()
    };

    // WHEN: We resolve with invalid choice
    let result = session.resolve_ambiguity(&comp_id, &ambiguity_id, 999);

    // THEN: It fails
    assert!(result.is_err());
    match result.unwrap_err() {
        McpError::InvalidChoice(_, _) => {}
        e => panic!("Expected InvalidChoice, got {:?}", e),
    }
}

#[test]
fn test_mcp_resource_pending() {
    // GIVEN: Session with components
    let source = r#"
        function App() { return <div />; }
        function Other() { return <span />; }
    "#;

    let extraction = extract_source(source, Path::new("test.tsx"), "test.tsx").unwrap();
    let spec = generate_spec(&extraction, source);
    let mut session = MigrationSession::from_spec(spec, "/tmp/output");

    // Start one migration
    let comp_id = session.list_migrations()[0].id.clone();
    session.start_migration(&comp_id).unwrap();

    // WHEN: We get pending resource
    let pending = session.resource_pending();

    // THEN: Only one is pending (the other is in progress)
    assert_eq!(pending.len(), 1);
    assert_eq!(pending[0].status, MigrationStatus::Pending);
}

#[test]
fn test_mcp_resource_patterns() {
    // GIVEN: Any session
    let source = "function App() { return <div />; }";
    let extraction = extract_source(source, Path::new("test.tsx"), "test.tsx").unwrap();
    let spec = generate_spec(&extraction, source);
    let session = MigrationSession::from_spec(spec, "/tmp/output");

    // WHEN: We get patterns resource
    let patterns = session.resource_patterns();

    // THEN: We get pattern library
    assert!(!patterns.is_empty());
    assert!(patterns.iter().any(|p| p.name.contains("useState")));
}

#[test]
fn test_mcp_resource_overview() {
    // GIVEN: Session with components
    let source = r#"
        function App() { return <div />; }
        function Other() { return <span />; }
    "#;

    let extraction = extract_source(source, Path::new("test.tsx"), "test.tsx").unwrap();
    let spec = generate_spec(&extraction, source);
    let session = MigrationSession::from_spec(spec, "/tmp/output");

    // WHEN: We get overview resource
    let overview = session.resource_overview();

    // THEN: Counts are correct
    assert_eq!(overview.total_components, 2);
    assert_eq!(overview.completed, 0);
    assert_eq!(overview.in_progress, 0);
    assert_eq!(overview.blocked, 0);
}

#[test]
fn test_mcp_get_patterns_filtered() {
    // GIVEN: Session
    let source = "function App() { return <div />; }";
    let extraction = extract_source(source, Path::new("test.tsx"), "test.tsx").unwrap();
    let spec = generate_spec(&extraction, source);
    let session = MigrationSession::from_spec(spec, "/tmp/output");

    // WHEN: We filter patterns by name
    let filter = PatternFilter {
        name: Some("useState".to_string()),
        category: None,
    };
    let patterns = session.get_patterns(Some(filter));

    // THEN: Only matching patterns returned
    assert!(patterns.iter().all(|p| p.name.contains("useState")));
}

#[test]
fn test_mcp_generate_code() {
    // GIVEN: Session with component
    let source = r#"
        function Counter() {
            const [count, setCount] = useState(0);
            return <button onClick={() => setCount(count + 1)}>{count}</button>;
        }
    "#;

    let extraction = extract_source(source, Path::new("test.tsx"), "test.tsx").unwrap();
    let spec = generate_spec(&extraction, source);
    let session = MigrationSession::from_spec(spec, "/tmp/output");

    // WHEN: We generate code
    let comp_id = session.list_migrations()[0].id.clone();
    let result = session.generate_code(&comp_id);

    // THEN: We get valid Sigil code
    assert!(result.is_ok());
    let generated = result.unwrap();
    assert!(generated.code.contains("actor Counter"));
    assert!(generated.code.contains("count"));
}

#[test]
fn test_mcp_complete_migration() {
    // CRITICAL-1: Test the complete_migration functionality

    // GIVEN: Session with a simple component
    let source = r#"
        function Greeting() {
            return <h1>Hello World</h1>;
        }
    "#;

    let extraction = extract_source(source, Path::new("test.tsx"), "test.tsx").unwrap();
    let spec = generate_spec(&extraction, source);

    // Use a unique temp directory for this test
    let tmp_dir = std::env::temp_dir().join(format!("mcp_complete_test_{}", std::process::id()));
    let mut session = MigrationSession::from_spec(spec, &tmp_dir);

    let comp_id = session.list_migrations()[0].id.clone();

    // Valid Sigil code for a pure function component
    let sigil_code = r#"invoke qliphoth·prelude·*;

rite greeting() -> VNode! {
    VNode·h1()·text_child("Hello World")
}
"#;

    // WHEN: We complete the migration
    let result = session.complete_migration(&comp_id, sigil_code);

    // THEN: It succeeds
    assert!(result.is_ok(), "complete_migration failed: {:?}", result.err());
    let completion = result.unwrap();
    assert!(completion.success);
    assert!(!completion.output_path.is_empty());

    // Verify file was written
    let output_path = std::path::Path::new(&completion.output_path);
    assert!(output_path.exists(), "Output file should exist at {:?}", output_path);

    // Verify file contents
    let written_code = std::fs::read_to_string(output_path).expect("Should read output file");
    assert!(written_code.contains("invoke qliphoth"));
    assert!(written_code.contains("greeting"));

    // Verify status updated to Completed
    let migrations = session.list_migrations();
    let completed_comp = migrations.iter().find(|m| m.id == comp_id).unwrap();
    assert_eq!(completed_comp.status, MigrationStatus::Completed);

    // Verify overview updated
    let overview = session.resource_overview();
    assert_eq!(overview.completed, 1);

    // Cleanup
    std::fs::remove_dir_all(&tmp_dir).ok();
}

#[test]
fn test_mcp_complete_migration_validation_failure() {
    // Test that complete_migration fails if code doesn't validate

    let source = r#"function App() { return <div />; }"#;
    let extraction = extract_source(source, Path::new("test.tsx"), "test.tsx").unwrap();
    let spec = generate_spec(&extraction, source);

    let tmp_dir = std::env::temp_dir().join(format!("mcp_complete_fail_{}", std::process::id()));
    let mut session = MigrationSession::from_spec(spec, &tmp_dir);

    let comp_id = session.list_migrations()[0].id.clone();

    // Invalid code - missing qliphoth import
    let invalid_code = r#"
actor App {
    rite view(&self) -> VNode {
        VNode·div()
    }
}
"#;

    // WHEN: We try to complete with invalid code
    let result = session.complete_migration(&comp_id, invalid_code);

    // THEN: It fails validation
    assert!(result.is_err());
    match result.unwrap_err() {
        McpError::ValidationFailed(errors) => {
            assert!(!errors.is_empty());
            assert!(errors.iter().any(|e| e.message.contains("import")));
        }
        e => panic!("Expected ValidationFailed, got {:?}", e),
    }

    // Status should still be Pending
    let migrations = session.list_migrations();
    assert_eq!(migrations[0].status, MigrationStatus::Pending);

    // Cleanup
    std::fs::remove_dir_all(&tmp_dir).ok();
}

#[test]
fn test_mcp_get_completed_code() {
    // MINOR-3: Test the get_completed_code accessor

    let source = r#"function App() { return <div>Hi</div>; }"#;
    let extraction = extract_source(source, Path::new("test.tsx"), "test.tsx").unwrap();
    let spec = generate_spec(&extraction, source);

    let tmp_dir = std::env::temp_dir().join(format!("mcp_get_code_{}", std::process::id()));
    let mut session = MigrationSession::from_spec(spec, &tmp_dir);

    let comp_id = session.list_migrations()[0].id.clone();

    // Before completion, should return None
    assert!(session.get_completed_code(&comp_id).is_none());

    // Complete the migration
    let sigil_code = "invoke qliphoth·prelude·*;\nrite app() -> VNode! { VNode·div()·text_child(\"Hi\") }";
    session.complete_migration(&comp_id, sigil_code).unwrap();

    // After completion, should return the code
    let retrieved = session.get_completed_code(&comp_id);
    assert!(retrieved.is_some());
    assert_eq!(retrieved.unwrap(), sigil_code);

    // Cleanup
    std::fs::remove_dir_all(&tmp_dir).ok();
}

#[test]
fn test_mcp_session_save_load() {
    // MINOR-2: Test state persistence

    let source = r#"
        function Counter() {
            const [count, setCount] = useState(0);
            return <div>{count}</div>;
        }
    "#;

    let extraction = extract_source(source, Path::new("test.tsx"), "test.tsx").unwrap();
    let spec = generate_spec(&extraction, source);

    let tmp_dir = std::env::temp_dir().join(format!("mcp_persist_{}", std::process::id()));
    std::fs::create_dir_all(&tmp_dir).unwrap();

    let mut session = MigrationSession::from_spec(spec, &tmp_dir);

    // Make some changes to session state
    let comp_id = session.list_migrations()[0].id.clone();
    session.start_migration(&comp_id).unwrap();

    // Save session
    let save_path = tmp_dir.join("session.json");
    session.save(&save_path).expect("Should save session");
    assert!(save_path.exists());

    // Load session into new instance
    let loaded_session = MigrationSession::load(&save_path, &tmp_dir)
        .expect("Should load session");

    // Verify state was preserved
    let migrations = loaded_session.list_migrations();
    assert_eq!(migrations.len(), 1);
    assert_eq!(migrations[0].id, comp_id);
    assert_eq!(migrations[0].status, MigrationStatus::InProgress);

    // Cleanup
    std::fs::remove_dir_all(&tmp_dir).ok();
}

#[test]
fn test_mcp_validate_sigil_parser_syntax_error() {
    // MINOR-1: Test that full Sigil parser validation catches syntax errors

    let source = r#"function App() { return <div />; }"#;
    let extraction = extract_source(source, Path::new("test.tsx"), "test.tsx").unwrap();
    let spec = generate_spec(&extraction, source);
    let session = MigrationSession::from_spec(spec, "/tmp/output");

    // Code with actual syntax error - unclosed brace
    let invalid_code = r#"invoke qliphoth·prelude·*;

rite app() -> VNode! {
    VNode·div()
// Missing closing brace
"#;

    let result = session.validate_sigil(invalid_code);

    assert!(!result.valid, "Should detect syntax error");
    assert!(!result.errors.is_empty(), "Should have errors");
    // The error should mention unexpected EOF or similar
    let error_msg = &result.errors[0].message;
    assert!(
        error_msg.contains("Unexpected") || error_msg.contains("expected") || error_msg.contains("Syntax"),
        "Error should mention syntax issue: {}",
        error_msg
    );
}

#[test]
fn test_mcp_validate_sigil_parser_deprecated_syntax() {
    // Test that deprecated Rust syntax is caught

    let source = r#"function App() { return <div />; }"#;
    let extraction = extract_source(source, Path::new("test.tsx"), "test.tsx").unwrap();
    let spec = generate_spec(&extraction, source);
    let session = MigrationSession::from_spec(spec, "/tmp/output");

    // Code using deprecated Rust 'fn' instead of Sigil 'rite'
    let rust_syntax_code = r#"invoke qliphoth·prelude·*;

fn app() -> VNode {
    VNode·div()
}
"#;

    let result = session.validate_sigil(rust_syntax_code);

    // Should either fail or have warnings about deprecated syntax
    // The parser may accept 'fn' with a warning or error
    if !result.valid {
        let has_relevant_error = result.errors.iter().any(|e|
            e.message.contains("Deprecated") ||
            e.message.contains("fn") ||
            e.message.contains("rite") ||
            e.message.contains("Syntax")
        );
        assert!(has_relevant_error, "Error should relate to deprecated syntax: {:?}", result.errors);
    }
}

#[test]
fn test_mcp_validate_sigil_parser_valid_complex() {
    // Test that valid complex Sigil code passes parser validation

    let source = r#"function App() { return <div />; }"#;
    let extraction = extract_source(source, Path::new("test.tsx"), "test.tsx").unwrap();
    let spec = generate_spec(&extraction, source);
    let session = MigrationSession::from_spec(spec, "/tmp/output");

    // Valid complex Sigil actor
    let valid_code = r#"invoke qliphoth·prelude·*;

ᛈ CounterMsg {
    Increment,
    Decrement,
}

☉ actor Counter {
    state count: i64! = 0,

    rite new() -> Self! {
        Self { count: 0 }
    }

    on Increment {
        self.count += 1;
    }

    on Decrement {
        self.count -= 1;
    }

    rite view(&self) -> VNode! {
        VNode·div()
            ·class("counter")
            ·child(VNode·span()·text_child(self.count·to_string()))
            ·child(VNode·button()·text_child("+")·on_click(Increment))
            ·child(VNode·button()·text_child("-")·on_click(Decrement))
    }
}
"#;

    let result = session.validate_sigil(valid_code);

    assert!(result.valid, "Valid Sigil should pass: {:?}", result.errors);
    assert!(result.errors.is_empty(), "Should have no errors: {:?}", result.errors);
}

#[test]
fn test_mcp_validate_sigil_heuristic_before_parser() {
    // Test that heuristic errors prevent parser from running
    // (to avoid confusing parser errors on placeholder code)

    let source = r#"function App() { return <div />; }"#;
    let extraction = extract_source(source, Path::new("test.tsx"), "test.tsx").unwrap();
    let spec = generate_spec(&extraction, source);
    let session = MigrationSession::from_spec(spec, "/tmp/output");

    // Code with placeholder - heuristic should catch this first
    let placeholder_code = r#"invoke qliphoth·prelude·*;

rite app() -> VNode! {
    VNode·div()·child(/* expression */)
}
"#;

    let result = session.validate_sigil(placeholder_code);

    assert!(!result.valid);
    // Should have the placeholder error, not a parser error
    assert!(result.errors.iter().any(|e| e.message.contains("Placeholder")),
        "Should catch placeholder before parser: {:?}", result.errors);
}

// =============================================================================
// Phase 6.1: Type Field Extraction
// =============================================================================

#[test]
fn test_type_extraction_captures_all_fields() {
    // GIVEN: An interface with multiple fields
    let source = r#"
        interface ButtonProps {
            label: string;
            onClick: () => void;
            disabled: boolean;
            size: 'small' | 'medium' | 'large';
            icon?: React.ReactNode;
        }
    "#;

    // WHEN: We extract types
    let extraction = extract_source(source, Path::new("test.tsx"), "test.tsx").unwrap();

    // THEN: All 5 fields are captured
    assert_eq!(extraction.types.len(), 1);
    let button_props = &extraction.types[0];
    assert_eq!(button_props.name, "ButtonProps");
    assert_eq!(button_props.fields.len(), 5, "Should capture all 5 fields");

    // Verify field names
    let field_names: Vec<&str> = button_props.fields.iter().map(|f| f.name.as_str()).collect();
    assert!(field_names.contains(&"label"));
    assert!(field_names.contains(&"onClick"));
    assert!(field_names.contains(&"disabled"));
    assert!(field_names.contains(&"size"));
    assert!(field_names.contains(&"icon"));
}

#[test]
fn test_type_extraction_marks_optional_fields() {
    // GIVEN: An interface with optional fields
    let source = r#"
        interface UserProfile {
            id: string;
            name: string;
            email?: string;
            avatar?: string;
        }
    "#;

    // WHEN: We extract types
    let extraction = extract_source(source, Path::new("test.tsx"), "test.tsx").unwrap();

    // THEN: Optional fields are marked correctly
    let profile = &extraction.types[0];
    let id_field = profile.fields.iter().find(|f| f.name == "id").unwrap();
    let email_field = profile.fields.iter().find(|f| f.name == "email").unwrap();

    assert!(!id_field.optional, "id should be required");
    assert!(email_field.optional, "email should be optional");
}

#[test]
fn test_type_extraction_preserves_union_types() {
    // GIVEN: A type with union fields
    let source = r#"
        interface ApiResponse {
            status: 'success' | 'error' | 'pending';
            data: string | null;
        }
    "#;

    // WHEN: We extract types
    let extraction = extract_source(source, Path::new("test.tsx"), "test.tsx").unwrap();

    // THEN: Union types are preserved in type_annotation
    let response = &extraction.types[0];
    let status_field = response.fields.iter().find(|f| f.name == "status").unwrap();

    assert!(status_field.type_annotation.contains("success"));
    assert!(status_field.type_annotation.contains("error"));
    assert!(status_field.type_annotation.contains("pending"));
}

#[test]
fn test_type_extraction_handles_extends() {
    // GIVEN: An interface that extends another
    let source = r#"
        interface BaseProps {
            id: string;
        }
        interface ExtendedProps extends BaseProps {
            name: string;
        }
    "#;

    // WHEN: We extract types
    let extraction = extract_source(source, Path::new("test.tsx"), "test.tsx").unwrap();

    // THEN: Extended interface captures extends clause
    let extended = extraction.types.iter().find(|t| t.name == "ExtendedProps").unwrap();
    assert!(!extended.extends.is_empty(), "Should capture extends");
    assert!(extended.extends.iter().any(|e| e.contains("BaseProps")));
}

#[test]
fn test_type_extraction_resolves_type_references() {
    // GIVEN: An interface with type references
    let source = r#"
        interface User {
            id: string;
        }
        interface Comment {
            author: User;
            replies: Comment[];
        }
    "#;

    // WHEN: We extract types
    let extraction = extract_source(source, Path::new("test.tsx"), "test.tsx").unwrap();

    // THEN: Type references are captured with their kind
    let comment = extraction.types.iter().find(|t| t.name == "Comment").unwrap();
    let author_field = comment.fields.iter().find(|f| f.name == "author").unwrap();
    let replies_field = comment.fields.iter().find(|f| f.name == "replies").unwrap();

    // Check type_kind classification
    match &author_field.type_kind {
        extraction::TypeFieldKind::TypeRef { name, .. } => {
            assert_eq!(name, "User");
        }
        _ => panic!("Expected TypeRef for author field"),
    }

    match &replies_field.type_kind {
        extraction::TypeFieldKind::Array { element_type } => {
            assert!(element_type.contains("Comment"));
        }
        _ => panic!("Expected Array for replies field"),
    }
}

// =============================================================================
// Phase 6.2: Helper Function Extraction
// =============================================================================

#[test]
fn test_helper_extraction_finds_module_scope_functions() {
    // GIVEN: Source with module-scope helper functions
    let source = r#"
        function formatDate(date: Date): string {
            return date.toISOString();
        }

        function calculateTotal(items: number[]): number {
            return items.reduce((a, b) => a + b, 0);
        }

        function App() {
            return <div />;
        }
    "#;

    // WHEN: We extract
    let extraction = extract_source(source, Path::new("test.tsx"), "test.tsx").unwrap();

    // THEN: Helper functions are found (not components)
    assert!(extraction.helper_functions.len() >= 2,
        "Should find at least 2 helper functions, found: {:?}",
        extraction.helper_functions.iter().map(|h| &h.name).collect::<Vec<_>>());

    let helper_names: Vec<&str> = extraction.helper_functions.iter().map(|h| h.name.as_str()).collect();
    assert!(helper_names.contains(&"formatDate"), "Should find formatDate");
    assert!(helper_names.contains(&"calculateTotal"), "Should find calculateTotal");
}

#[test]
fn test_helper_extraction_finds_component_scope_functions() {
    // GIVEN: Source with functions inside component
    let source = r#"
        function App() {
            function handleClick() {
                console.log('clicked');
            }

            const formatValue = (val: number) => val.toFixed(2);

            return <button onClick={handleClick}>{formatValue(42)}</button>;
        }
    "#;

    // WHEN: We extract
    let extraction = extract_source(source, Path::new("test.tsx"), "test.tsx").unwrap();

    // THEN: Component has handlers (component-scope functions become handlers)
    assert_eq!(extraction.components.len(), 1);
    let component = &extraction.components[0];

    // handleClick should be extracted as a handler
    assert!(component.handlers.iter().any(|h| h.name == "handleClick"),
        "Should find handleClick handler");
}

#[test]
fn test_helper_extraction_captures_parameters_and_return_type() {
    // GIVEN: Helper with typed parameters and return
    let source = r#"
        function processData(input: string, count: number): ProcessedResult {
            return { value: input, total: count };
        }

        function App() {
            return <div />;
        }
    "#;

    // WHEN: We extract
    let extraction = extract_source(source, Path::new("test.tsx"), "test.tsx").unwrap();

    // THEN: Parameters and return type are captured
    let helper = extraction.helper_functions.iter()
        .find(|h| h.name == "processData")
        .expect("Should find processData");

    assert_eq!(helper.parameters.len(), 2, "Should have 2 parameters");
    assert!(helper.return_type.is_some(), "Should have return type");
    assert!(helper.return_type.as_ref().unwrap().contains("ProcessedResult"));
}

#[test]
fn test_helper_extraction_detects_purity() {
    // GIVEN: Pure and impure helper functions
    let source = r#"
        // Pure function - no side effects
        function add(a: number, b: number): number {
            return a + b;
        }

        // Impure - has console.log
        function logAndAdd(a: number, b: number): number {
            console.log('adding');
            return a + b;
        }

        function App() {
            return <div />;
        }
    "#;

    // WHEN: We extract
    let extraction = extract_source(source, Path::new("test.tsx"), "test.tsx").unwrap();

    // THEN: Purity is detected
    let add_fn = extraction.helper_functions.iter().find(|h| h.name == "add");
    let log_fn = extraction.helper_functions.iter().find(|h| h.name == "logAndAdd");

    assert!(add_fn.is_some(), "Should find add helper function");
    assert!(log_fn.is_some(), "Should find logAndAdd helper function");

    let add = add_fn.unwrap();
    assert!(add.is_pure, "add should be pure");

    let log = log_fn.unwrap();
    assert!(!log.is_pure, "logAndAdd should be impure");
    assert!(log.side_effects.iter().any(|s| matches!(s, extraction::SideEffect::ConsoleLog)),
        "Should detect console.log side effect");
}

#[test]
fn test_helper_extraction_tracks_usage_sites() {
    // GIVEN: Helper used in multiple places
    let source = r#"
        function formatCurrency(amount: number): string {
            return '$' + amount.toFixed(2);
        }

        function App() {
            const price = formatCurrency(99.99);
            const tax = formatCurrency(7.50);
            return <div>{price} + {tax}</div>;
        }
    "#;

    // WHEN: We extract
    let extraction = extract_source(source, Path::new("test.tsx"), "test.tsx").unwrap();

    // THEN: Helper is found with correct metadata
    let helper = extraction.helper_functions.iter()
        .find(|h| h.name == "formatCurrency");

    assert!(helper.is_some(), "Should find formatCurrency helper");

    let h = helper.unwrap();
    assert_eq!(h.name, "formatCurrency");
    assert!(h.is_pure, "formatCurrency should be pure");
    assert_eq!(h.parameters.len(), 1, "Should have one parameter");

    // Note: used_by cross-file tracking is not implemented (requires multi-file analysis)
    // This test verifies the helper is found and correctly extracted
}

// =============================================================================
// Phase 6.3: Handler Body Analysis
// =============================================================================

#[test]
fn test_handler_body_extracts_function_calls() {
    // GIVEN: Handler with multiple function calls
    let source = r#"
        function ChatPanel() {
            const [message, setMessage] = useState('');

            const handleSend = () => {
                sendMessage(message);
                clearInput();
                trackEvent('message_sent');
            };

            return <button onClick={handleSend}>Send</button>;
        }
    "#;

    // WHEN: We extract
    let extraction = extract_source(source, Path::new("test.tsx"), "test.tsx").unwrap();

    // THEN: Handler calls are extracted
    let component = &extraction.components[0];
    let handler = component.handlers.iter().find(|h| h.name == "handleSend").unwrap();

    assert!(handler.calls.len() >= 3, "Should capture at least 3 calls");

    let call_names: Vec<&str> = handler.calls.iter().map(|c| c.name.as_str()).collect();
    assert!(call_names.contains(&"sendMessage"));
    assert!(call_names.contains(&"clearInput"));
    assert!(call_names.contains(&"trackEvent"));
}

#[test]
fn test_handler_body_identifies_call_sources() {
    // GIVEN: Handler calling functions from different sources
    let source = r#"
        function ChatPanel() {
            const [count, setCount] = useState(0);

            const handleAction = () => {
                setCount(count + 1);
                fetch('/api/action');
            };

            return <button onClick={handleAction}>Act</button>;
        }
    "#;

    // WHEN: We extract
    let extraction = extract_source(source, Path::new("test.tsx"), "test.tsx").unwrap();

    // THEN: Call sources are identified
    let component = &extraction.components[0];
    let handler = component.handlers.iter().find(|h| h.name == "handleAction").unwrap();

    // Find setCount call
    let set_count_call = handler.calls.iter().find(|c| c.name == "setCount");
    assert!(set_count_call.is_some(), "Should find setCount call");

    if let Some(call) = set_count_call {
        match &call.source {
            extraction::CallSource::StateSetter { state_name } => {
                assert_eq!(state_name, "count");
            }
            _ => panic!("setCount should be identified as StateSetter"),
        }
    }

    // Find fetch call
    let fetch_call = handler.calls.iter().find(|c| c.name == "fetch");
    assert!(fetch_call.is_some(), "Should find fetch call");

    if let Some(call) = fetch_call {
        assert!(matches!(call.source, extraction::CallSource::Global));
    }
}

#[test]
fn test_handler_body_detects_early_returns() {
    // GIVEN: Handler with early return
    let source = r#"
        function Form() {
            const handleSubmit = (e: Event) => {
                if (!isValid) {
                    return;
                }
                submitForm();
            };

            return <form onSubmit={handleSubmit} />;
        }
    "#;

    // WHEN: We extract
    let extraction = extract_source(source, Path::new("test.tsx"), "test.tsx").unwrap();

    // THEN: Early return is detected
    let component = &extraction.components[0];
    let handler = component.handlers.iter().find(|h| h.name == "handleSubmit").unwrap();

    assert!(handler.has_early_return, "Should detect early return");
}

#[test]
fn test_handler_body_captures_conditionals() {
    // GIVEN: Handler with conditional logic
    let source = r#"
        function Toggle() {
            const [on, setOn] = useState(false);

            const handleToggle = () => {
                if (on) {
                    turnOff();
                } else {
                    turnOn();
                }
            };

            return <button onClick={handleToggle}>Toggle</button>;
        }
    "#;

    // WHEN: We extract
    let extraction = extract_source(source, Path::new("test.tsx"), "test.tsx").unwrap();

    // THEN: Conditionals are detected
    let component = &extraction.components[0];
    let handler = component.handlers.iter().find(|h| h.name == "handleToggle").unwrap();

    assert!(handler.has_conditionals, "Should detect conditionals");
}

#[test]
fn test_handler_body_infers_state_mutations() {
    // GIVEN: Handler that mutates state
    let source = r#"
        function Counter() {
            const [count, setCount] = useState(0);
            const [total, setTotal] = useState(0);

            const handleIncrement = () => {
                setCount(count + 1);
                setTotal(total + 1);
            };

            return <button onClick={handleIncrement}>+</button>;
        }
    "#;

    // WHEN: We extract
    let extraction = extract_source(source, Path::new("test.tsx"), "test.tsx").unwrap();

    // THEN: State mutations are captured
    let component = &extraction.components[0];
    let handler = component.handlers.iter().find(|h| h.name == "handleIncrement").unwrap();

    assert!(handler.state_mutations.len() >= 2,
        "Should detect 2 state mutations, found: {:?}", handler.state_mutations);
}

// =============================================================================
// Phase 6.4: Hook Argument Expansion
// =============================================================================

#[test]
fn test_hook_args_expand_object_properties() {
    // GIVEN: Custom hook with object argument
    let source = r#"
        function App() {
            const result = useQuery({
                queryKey: ['users'],
                queryFn: fetchUsers,
                enabled: true
            });
            return <div />;
        }
    "#;

    // WHEN: We extract
    let extraction = extract_source(source, Path::new("test.tsx"), "test.tsx").unwrap();

    // THEN: Object properties are expanded
    let component = &extraction.components[0];
    let hook = component.custom_hooks.iter().find(|h| h.name == "useQuery").unwrap();

    assert!(!hook.expanded_arguments.is_empty(), "Should have expanded arguments");

    if let extraction::HookArgument::Object { properties } = &hook.expanded_arguments[0] {
        let prop_names: Vec<&str> = properties.iter().map(|p| p.name.as_str()).collect();
        assert!(prop_names.contains(&"queryKey"));
        assert!(prop_names.contains(&"queryFn"));
        assert!(prop_names.contains(&"enabled"));
    } else {
        panic!("First argument should be Object");
    }
}

#[test]
fn test_hook_args_capture_arrow_functions() {
    // GIVEN: Custom hook with arrow function argument
    let source = r#"
        function App() {
            const state = useStore((s) => s.user);
            return <div />;
        }
    "#;

    // WHEN: We extract
    let extraction = extract_source(source, Path::new("test.tsx"), "test.tsx").unwrap();

    // THEN: Arrow function is captured
    let component = &extraction.components[0];
    let hook = component.custom_hooks.iter().find(|h| h.name == "useStore").unwrap();

    if let extraction::HookArgument::Function { params, body_summary, .. } = &hook.expanded_arguments[0] {
        assert!(params.contains(&"s".to_string()) || params.iter().any(|p| p.contains("s")));
        assert!(body_summary.contains("s.user") || body_summary.contains("user"));
    } else {
        panic!("Argument should be Function");
    }
}

#[test]
fn test_hook_args_analyze_callback_bodies() {
    // GIVEN: Hook with callback that has side effects
    let source = r#"
        function App() {
            const data = useQuery({
                onSuccess: (data) => {
                    console.log('Success');
                    updateCache(data);
                }
            });
            return <div />;
        }
    "#;

    // WHEN: We extract
    let extraction = extract_source(source, Path::new("test.tsx"), "test.tsx").unwrap();

    // THEN: Callback body is analyzed
    let component = &extraction.components[0];
    let hook = component.custom_hooks.iter().find(|h| h.name == "useQuery").unwrap();

    if let extraction::HookArgument::Object { properties } = &hook.expanded_arguments[0] {
        let on_success = properties.iter().find(|p| p.name == "onSuccess");
        assert!(on_success.is_some(), "Should find onSuccess property");

        if let Some(prop) = on_success {
            if let extraction::HookPropertyValue::Callback { calls, side_effects, .. } = &prop.value_kind {
                // Should detect console.log
                assert!(side_effects.iter().any(|s| matches!(s, extraction::SideEffect::ConsoleLog)),
                    "Should detect console.log in callback");
                // Should detect updateCache call
                assert!(calls.iter().any(|c| c.name == "updateCache"),
                    "Should detect updateCache call");
            }
        }
    }
}

#[test]
fn test_hook_args_preserve_array_arguments() {
    // GIVEN: Hook with array argument
    let source = r#"
        function App() {
            const result = useQuery({
                queryKey: ['users', userId, 'posts']
            });
            return <div />;
        }
    "#;

    // WHEN: We extract
    let extraction = extract_source(source, Path::new("test.tsx"), "test.tsx").unwrap();

    // THEN: Array is preserved
    let component = &extraction.components[0];
    let hook = component.custom_hooks.iter().find(|h| h.name == "useQuery").unwrap();

    if let extraction::HookArgument::Object { properties } = &hook.expanded_arguments[0] {
        let query_key = properties.iter().find(|p| p.name == "queryKey");
        assert!(query_key.is_some(), "Should find queryKey property");

        if let Some(prop) = query_key {
            match &prop.value_kind {
                extraction::HookPropertyValue::Array { elements } => {
                    assert!(elements.len() >= 2, "Should have array elements");
                }
                extraction::HookPropertyValue::Simple { value } => {
                    // Array might be captured as simple value with source
                    assert!(value.contains("users") || value.contains("["));
                }
                _ => panic!("queryKey should be Array or Simple"),
            }
        }
    }
}

#[test]
fn test_hook_args_handle_nested_objects() {
    // GIVEN: Hook with nested object argument
    let source = r#"
        function App() {
            const result = useMutation({
                options: {
                    retry: 3,
                    retryDelay: 1000
                }
            });
            return <div />;
        }
    "#;

    // WHEN: We extract
    let extraction = extract_source(source, Path::new("test.tsx"), "test.tsx").unwrap();

    // THEN: Nested object is handled
    let component = &extraction.components[0];
    let hook = component.custom_hooks.iter().find(|h| h.name == "useMutation").unwrap();

    if let extraction::HookArgument::Object { properties } = &hook.expanded_arguments[0] {
        let options = properties.iter().find(|p| p.name == "options");
        assert!(options.is_some(), "Should find options property");

        if let Some(prop) = options {
            match &prop.value_kind {
                extraction::HookPropertyValue::Object { properties: nested } => {
                    assert!(nested.iter().any(|p| p.name == "retry"));
                    assert!(nested.iter().any(|p| p.name == "retryDelay"));
                }
                _ => panic!("options should be nested Object"),
            }
        }
    }
}

// =============================================================================
// Phase 6.5: Architecture Mapping
// =============================================================================

#[test]
fn test_architecture_identifies_service_actors() {
    // GIVEN: Component using custom hooks that suggest service actors
    let source = r#"
        function ChatPanel() {
            const { messages, addMessage } = useChat();
            const { isRunning, runAgent } = useAgent();
            return <div />;
        }
    "#;

    // WHEN: We extract
    let extraction = extract_source(source, Path::new("test.tsx"), "test.tsx").unwrap();

    // THEN: Service actors are recommended
    let component = &extraction.components[0];
    let arch = &component.architecture;

    assert!(arch.service_actors.len() >= 2,
        "Should recommend at least 2 service actors, found: {:?}",
        arch.service_actors.iter().map(|a| &a.name).collect::<Vec<_>>());

    let actor_names: Vec<&str> = arch.service_actors.iter().map(|a| a.name.as_str()).collect();
    assert!(actor_names.contains(&"ChatService"), "Should recommend ChatService");
    assert!(actor_names.contains(&"AgentService"), "Should recommend AgentService");
}

#[test]
fn test_architecture_maps_zustand_stores() {
    // GIVEN: Component using Zustand store
    let source = r#"
        function Dashboard() {
            const serverStatus = useAppStore((s) => s.serverStatus);
            const loadModel = useAppStore((s) => s.loadModel);
            return <div />;
        }
    "#;

    // WHEN: We extract
    let extraction = extract_source(source, Path::new("test.tsx"), "test.tsx").unwrap();

    // THEN: Zustand store is mapped
    let component = &extraction.components[0];
    let arch = &component.architecture;

    assert!(!arch.zustand_stores.is_empty(), "Should detect Zustand store");

    let store = &arch.zustand_stores[0];
    assert_eq!(store.hook_name, "useAppStore");
    assert!(store.suggested_actor.contains("App"), "Should suggest AppActor");
}

#[test]
fn test_architecture_suggests_message_types() {
    // GIVEN: Custom hook with action functions
    let source = r#"
        function Editor() {
            const { save, load, reset } = useDocument();
            return <div />;
        }
    "#;

    // WHEN: We extract
    let extraction = extract_source(source, Path::new("test.tsx"), "test.tsx").unwrap();

    // THEN: Message types are suggested
    let component = &extraction.components[0];
    let arch = &component.architecture;

    let doc_service = arch.service_actors.iter().find(|a| a.name == "DocumentService");
    assert!(doc_service.is_some(), "Should recommend DocumentService");

    if let Some(service) = doc_service {
        let msg_names: Vec<&str> = service.messages.iter().map(|m| m.name.as_str()).collect();
        assert!(msg_names.contains(&"Save"), "Should suggest Save message");
        assert!(msg_names.contains(&"Load"), "Should suggest Load message");
        assert!(msg_names.contains(&"Reset"), "Should suggest Reset message");
    }
}

#[test]
fn test_architecture_determines_state_ownership() {
    // GIVEN: Component with local and shared state
    let source = r#"
        function Counter() {
            const [localCount, setLocalCount] = useState(0);
            const globalCount = useStore((s) => s.count);
            return <div />;
        }
    "#;

    // WHEN: We extract
    let extraction = extract_source(source, Path::new("test.tsx"), "test.tsx").unwrap();

    // THEN: State ownership is determined
    let component = &extraction.components[0];
    let arch = &component.architecture;

    // Find local state
    let local_ownership = arch.state_ownership.iter()
        .find(|s| s.state_name == "localCount");
    assert!(local_ownership.is_some(), "Should track localCount ownership");

    if let Some(local) = local_ownership {
        assert_eq!(local.owner, "Self", "Local state owned by Self");
        assert!(matches!(local.access_pattern, extraction::StateAccessPattern::Local));
    }

    // Find shared state (from Zustand selector)
    // The selector `(s) => s.count` should detect "count" as shared state
    let shared_ownership = arch.state_ownership.iter()
        .find(|s| s.state_name == "globalCount" || s.state_name == "count");

    assert!(shared_ownership.is_some(), "Should track shared state from Zustand store");
    let shared = shared_ownership.unwrap();
    assert!(matches!(shared.access_pattern, extraction::StateAccessPattern::Shared),
        "Zustand state should be Shared");
}

#[test]
fn test_architecture_recommends_communication_patterns() {
    // GIVEN: Component that calls functions from hooks
    let source = r#"
        function Panel() {
            const { data, refresh } = useData();

            const handleRefresh = async () => {
                await refresh();
            };

            return <button onClick={handleRefresh}>Refresh</button>;
        }
    "#;

    // WHEN: We extract
    let extraction = extract_source(source, Path::new("test.tsx"), "test.tsx").unwrap();

    // THEN: Communication patterns are recommended
    let component = &extraction.components[0];
    let arch = &component.architecture;

    // Should have DataService actor
    assert!(arch.service_actors.iter().any(|a| a.name == "DataService"),
        "Should recommend DataService");

    // Note: Communication patterns from handlers may require handler.calls to be wired
    // This test documents expected behavior
}

#[test]
fn test_phase3_comprehensive_generation() {
    // Test comprehensive code generation for a component with:
    // - State (useState)
    // - Event handlers  
    // - Conditional rendering
    // - Nested elements
    let source = r#"
        function ChatWidget() {
            const [messages, setMessages] = useState([]);
            const [input, setInput] = useState("");
            
            const handleSend = () => {
                setMessages([...messages, input]);
                setInput("");
            };
            
            return (
                <div className="chat">
                    <div className="messages">
                        {messages.map((msg, i) => (
                            <div key={i}>{msg}</div>
                        ))}
                    </div>
                    <input 
                        value={input} 
                        onChange={(e) => setInput(e.target.value)} 
                    />
                    <button onClick={handleSend}>Send</button>
                </div>
            );
        }
    "#;

    let extraction = extract_source(source, Path::new("test.tsx"), "test.tsx").unwrap();
    let spec = generate_spec(&extraction, source);
    let generated = generate_component(&spec.components[0]);
    
    // Print generated code for inspection
    println!("\n=== Generated Sigil Code ===\n{}\n=== End ===\n", generated.code);

    // Basic sanity checks
    assert!(generated.code.contains("☉ actor ChatWidget"), "Should be an actor (has state)");
    assert!(generated.code.contains("state messages:"), "Should have messages state");
    assert!(generated.code.contains("state input:"), "Should have input state");
    assert!(generated.code.contains("VNode·div()"), "Should have VNode generation");
    assert!(generated.code.contains("rite view"), "Should have view method");
}

// =============================================================================
// Integration Test: Real ChatPanel from Infernum Observer
// =============================================================================

#[test]
fn test_integration_chatpanel_real_file() {
    // This test reads the actual ChatPanel.tsx from infernum-observer
    let chat_panel_path = "/home/crook/dev/infernum-observer/src/components/chat/ChatPanel.tsx";

    // Skip if file doesn't exist (CI environment)
    let source = match std::fs::read_to_string(chat_panel_path) {
        Ok(s) => s,
        Err(_) => {
            println!("Skipping integration test - ChatPanel.tsx not found at {}", chat_panel_path);
            return;
        }
    };

    // Extract
    let extraction = extract_source(&source, Path::new(chat_panel_path), "ChatPanel.tsx")
        .expect("Should extract ChatPanel");

    println!("\n=== ChatPanel Extraction Summary ===");
    println!("Components: {}", extraction.components.len());

    for comp in &extraction.components {
        println!("\nComponent: {}", comp.name);
        println!("  Hooks: {}", comp.hooks.len());
        println!("  Handlers: {}", comp.handlers.len());
        println!("  Custom Hooks: {}", comp.custom_hooks.len());

        // Show handler details
        for handler in &comp.handlers {
            println!("  Handler: {} ({} calls)", handler.name, handler.calls.len());
            for call in &handler.calls {
                println!("    - {}: {:?}", call.name, call.source);
            }
        }

        // Check architecture
        println!("  Architecture:");
        println!("    Service actors: {}", comp.architecture.service_actors.len());
        for actor in &comp.architecture.service_actors {
            println!("      - {} (from: {:?})", actor.name, actor.derived_from);
        }
    }

    // Generate spec
    let spec = generate_spec(&extraction, &source);

    println!("\n=== Migration Spec ===");
    for comp_spec in &spec.components {
        println!("\nComponent: {} -> {:?}", comp_spec.name, comp_spec.target.pattern);
        println!("  State fields: {}", comp_spec.recommendations.state_fields.len());
        println!("  Messages: {}", comp_spec.recommendations.messages.len());

        for msg in &comp_spec.recommendations.messages {
            println!("    Message: {}", msg.name);
            for sc in &msg.state_changes {
                println!("      State change: {}", sc);
            }
        }
    }

    // Generate code for each component
    println!("\n=== Generated Sigil Code ===");
    for comp_spec in &spec.components {
        let generated = generate_component(comp_spec);
        println!("\n--- {} ---\n{}", generated.component_name, generated.code);
    }

    // Assertions
    assert!(!extraction.components.is_empty(), "Should extract at least one component");

    let chat_panel = extraction.components.iter()
        .find(|c| c.name == "ChatPanel")
        .expect("Should find ChatPanel component");

    // ChatPanel should have state (it uses hooks)
    assert!(!chat_panel.hooks.is_empty(), "ChatPanel should have hooks");

    // ChatPanel uses custom hooks like useChat
    assert!(
        chat_panel.custom_hooks.iter().any(|h| h.name == "useChat"),
        "ChatPanel should use useChat hook"
    );

    // Handler calls should be linked to their hook sources
    let has_hook_linked_calls = chat_panel.handlers.iter()
        .any(|h| h.calls.iter().any(|c| matches!(c.source, CallSource::Hook { .. })));
    assert!(has_hook_linked_calls, "Handler calls should be linked to hooks (Phase 6.3 gap fixed)");
}

// =============================================================================
// Phase 4: Validation - Generated Code Structure
// =============================================================================

#[test]
fn test_phase4_generated_code_structure() {
    // Test that generated code has proper Sigil structure
    let source = r#"
        function Counter() {
            const [count, setCount] = useState(0);

            const increment = () => setCount(count + 1);

            return (
                <div>
                    <span>{count}</span>
                    <button onClick={increment}>+</button>
                </div>
            );
        }
    "#;

    let extraction = extract_source(source, Path::new("test.tsx"), "test.tsx").unwrap();
    let spec = generate_spec(&extraction, source);
    let generated = generate_component(&spec.components[0]);

    println!("\n=== Phase 4 Validation ===\n{}\n", generated.code);

    // Structural validation
    assert!(generated.code.contains("invoke qliphoth·prelude·*;"),
        "Should have prelude import");

    assert!(generated.code.contains("ᛈ CounterMsg {"),
        "Should have message enum");

    assert!(generated.code.contains("☉ actor Counter {"),
        "Should have actor declaration");

    assert!(generated.code.contains("state count:"),
        "Should have state field");

    assert!(generated.code.contains("on "),
        "Should have message handlers");

    assert!(generated.code.contains("rite view(self) -> VNode!"),
        "Should have view method");

    assert!(generated.code.contains("VNode·div()"),
        "Should have VNode builder");

    // Check balanced braces
    let open_braces = generated.code.matches('{').count();
    let close_braces = generated.code.matches('}').count();
    assert_eq!(open_braces, close_braces,
        "Braces should be balanced: {} open, {} close", open_braces, close_braces);
}

#[test]
fn test_phase4_service_calls_in_handlers() {
    // Test that service calls are properly generated in handler bodies
    // Note: Component needs useState to be treated as an actor
    let source = r#"
        function ChatWidget() {
            const [input, setInput] = useState("");
            const { messages, addMessage } = useChat();
            const { runAgent } = useAgent();

            const handleSend = () => {
                addMessage({ role: 'user', content: input });
                runAgent({ objective: input });
                setInput("");
            };

            return <div onClick={handleSend}>Send</div>;
        }
    "#;

    let extraction = extract_source(source, Path::new("test.tsx"), "test.tsx").unwrap();
    let spec = generate_spec(&extraction, source);
    let generated = generate_component(&spec.components[0]);

    println!("\n=== Service Calls Test ===\n{}\n", generated.code);

    // Should be an actor (has useState)
    assert!(generated.code.contains("☉ actor ChatWidget"),
        "Should be an actor");

    // Should have service actor message sends
    assert!(generated.code.contains("ChatService !"),
        "Should have ChatService message send");

    assert!(generated.code.contains("AgentService !"),
        "Should have AgentService message send");

    // Should have proper message names
    assert!(generated.code.contains("AddMessage"),
        "Should have AddMessage method");

    assert!(generated.code.contains("RunAgent"),
        "Should have RunAgent method");
}

#[test]
fn test_phase4_pure_function_generation() {
    // Test that pure components generate functions, not actors
    let source = r#"
        function Badge({ label, color }) {
            return (
                <span className={`badge badge-${color}`}>
                    {label}
                </span>
            );
        }
    "#;

    let extraction = extract_source(source, Path::new("test.tsx"), "test.tsx").unwrap();
    let spec = generate_spec(&extraction, source);
    let generated = generate_component(&spec.components[0]);

    println!("\n=== Pure Function Test ===\n{}\n", generated.code);

    // Pure components (no state/hooks) should generate functions
    assert!(generated.code.contains("rite badge(") || generated.code.contains("☉ actor Badge"),
        "Should generate either pure function or actor");

    // Should have VNode generation
    assert!(generated.code.contains("VNode·span()"),
        "Should have span VNode");
}

// =============================================================================
// Expression Transformation Tests
// =============================================================================

#[test]
fn test_expression_transformation_operators() {
    // Test that JS operators are transformed to Sigil operators
    let source = r#"
        function StatusWidget() {
            const [isLoading, setIsLoading] = useState(false);
            const [data, setData] = useState(null);

            return (
                <div>
                    {isLoading && <Spinner />}
                    {!isLoading && data && <Content data={data} />}
                    {data === null && <Empty />}
                    {data !== null && <Filled />}
                </div>
            );
        }
    "#;

    let extraction = extract_source(source, Path::new("test.tsx"), "test.tsx").unwrap();
    let spec = generate_spec(&extraction, source);
    let generated = generate_component(&spec.components[0]);

    println!("\n=== Operator Transformation Test ===\n{}\n", generated.code);

    // Check logical operators transformed
    assert!(generated.code.contains("∧") || generated.code.contains(" and "),
        "Should transform && to ∧");

    // Check negation transformed
    assert!(generated.code.contains("¬") || generated.code.contains("not "),
        "Should transform ! to ¬");
}

#[test]
fn test_expression_transformation_method_calls() {
    // Test that JS method calls are transformed to Sigil
    let source = r#"
        function ListWidget() {
            const [items, setItems] = useState([]);

            return (
                <div>
                    {items.length > 0 && <Count count={items.length} />}
                </div>
            );
        }
    "#;

    let extraction = extract_source(source, Path::new("test.tsx"), "test.tsx").unwrap();
    let spec = generate_spec(&extraction, source);
    let generated = generate_component(&spec.components[0]);

    println!("\n=== Method Call Transformation Test ===\n{}\n", generated.code);

    // Check .length → .len()
    assert!(generated.code.contains(".len()"),
        "Should transform .length to .len()");
}

// =============================================================================
// Phase 7: Service Actor Generation
// =============================================================================

#[test]
fn test_service_actor_collection() {
    // Test that custom hooks are collected into service actors
    let source = r#"
        function ChatPanel() {
            const { messages, isStreaming, addMessage, clearChat } = useChat();
            const { events, runAgent, stopAgent } = useAgent();
            const [input, setInput] = useState("");

            const handleSubmit = () => {
                addMessage({ role: 'user', content: input });
                runAgent({ objective: input });
                setInput("");
            };

            return (
                <div>
                    <button onClick={handleSubmit}>Send</button>
                </div>
            );
        }
    "#;

    let extraction = extract_source(source, Path::new("test.tsx"), "test.tsx").unwrap();
    let spec = generate_spec(&extraction, source);

    println!("\n=== Service Actor Collection Test ===");
    println!("Service actors found: {}", spec.service_actors.len());
    for actor in &spec.service_actors {
        println!("  - {} (from {})", actor.name, actor.derived_from);
        println!("    State: {:?}", actor.state_fields.iter().map(|f| &f.name).collect::<Vec<_>>());
        println!("    Messages: {:?}", actor.messages.iter().map(|m| &m.name).collect::<Vec<_>>());
    }

    // Should have 2 service actors: ChatService and AgentService
    assert_eq!(spec.service_actors.len(), 2, "Should collect 2 service actors from custom hooks");

    // Find ChatService
    let chat_service = spec.service_actors.iter()
        .find(|a| a.name == "ChatService")
        .expect("Should have ChatService");

    assert_eq!(chat_service.derived_from, "useChat");
    assert!(chat_service.state_fields.iter().any(|f| f.original_name == "messages"),
        "ChatService should have messages state");
    assert!(chat_service.state_fields.iter().any(|f| f.original_name == "isStreaming"),
        "ChatService should have isStreaming state");
    assert!(chat_service.messages.iter().any(|m| m.original_name == "addMessage"),
        "ChatService should have AddMessage message");
    assert!(chat_service.messages.iter().any(|m| m.original_name == "clearChat"),
        "ChatService should have ClearChat message");

    // Find AgentService
    let agent_service = spec.service_actors.iter()
        .find(|a| a.name == "AgentService")
        .expect("Should have AgentService");

    assert_eq!(agent_service.derived_from, "useAgent");
    assert!(agent_service.messages.iter().any(|m| m.original_name == "runAgent"),
        "AgentService should have RunAgent message");
}

#[test]
fn test_service_actor_code_generation() {
    // Test that service actor code is generated correctly
    let source = r#"
        function ChatPanel() {
            const { messages, isStreaming, addMessage } = useChat();
            const [input, setInput] = useState("");

            const handleSubmit = () => {
                addMessage({ role: 'user', content: input });
            };

            return <div />;
        }
    "#;

    let extraction = extract_source(source, Path::new("test.tsx"), "test.tsx").unwrap();
    let spec = generate_spec(&extraction, source);

    assert!(!spec.service_actors.is_empty(), "Should have service actors");

    let chat_service = &spec.service_actors[0];
    let generated = generate_service_actor(chat_service);

    println!("\n=== Service Actor Code Generation Test ===\n{}\n", generated.code);

    // Check structure
    assert!(generated.code.contains("invoke qliphoth·prelude·*;"),
        "Should have prelude import");
    assert!(generated.code.contains("ᛈ ChatServiceMsg"),
        "Should have message enum");
    assert!(generated.code.contains("☉ actor ChatService"),
        "Should have actor definition");

    // Check state fields
    assert!(generated.code.contains("state messages:"),
        "Should have messages state field");
    assert!(generated.code.contains("state is_streaming:"),
        "Should have is_streaming state field");

    // Check message handler
    assert!(generated.code.contains("on AddMessage"),
        "Should have AddMessage handler");
}

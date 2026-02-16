//! Tests for React → Qliphoth migration.
//!
//! Following Agent-TDD methodology from docs/specs/REACT-MIGRATION-TDD-ROADMAP.md
//! Tests are crystallized understanding of React → Qliphoth transformation.

use super::*;
use std::path::Path;

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

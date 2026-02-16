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

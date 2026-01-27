# Sigil Standard Library Reference

The standard library (`std`) provides core functionality for Sigil programs.

---

## Modules Overview

| Module | Description |
|--------|-------------|
| `std·io` | Input/output operations |
| `std·fs` | File system operations |
| `std·collections` | Data structures (Vec, HashMap, etc.) |
| `std·string` | String manipulation |
| `std·http` | HTTP client |
| `std·json` | JSON parsing and serialization |
| `std·time` | Date and time |
| `std·async` | Async/await primitives |
| `std·mem` | Memory management (Rc, Cell, Drop) |

---

## std·io

### Print Functions

```sigil
print("no newline");
println("with newline");
eprintln("to stderr");
```

### Reading Input

```sigil
use std·io;

≔ line = io·read_line()?;
≔ all = io·read_to_string()?;
```

---

## std·fs

### File Operations

```sigil
use std·fs;

// Read entire file
≔ content = fs·read_to_string("path/to/file.txt")?;

// Read as bytes
≔ bytes = fs·read("path/to/file.bin")?;

// Write file
fs·write("path/to/file.txt", "content")?;

// Append to file
fs·append("path/to/file.txt", "more content")?;

// Check existence
if fs·exists("path/to/file") {
    // ...
}

// Create directory
fs·create_dir("path/to/dir")?;
fs·create_dir_all("path/to/nested/dir")?;

// Remove
fs·remove_file("path/to/file")?;
fs·remove_dir("path/to/dir")?;

// List directory
≔ entries = fs·read_dir("path/to/dir")?;
∀ entry ∈ entries {
    println(entry.path);
}
```

---

## std·collections

### Vec (Dynamic Array)

```sigil
use std·collections·Vec;

≔ vec! = Vec·new();
vec.push(1);
vec.push(2);
vec.push(3);

≔ first = vec[0];        // Access by index
≔ len = vec.len();       // Length
≔ popped = vec.pop();    // Remove last

// Iteration
∀ item ∈ vec {
    println(str(item));
}

// Functional methods
≔ doubled = vec.iter().map(|x| x * 2).collect();
≔ evens = vec.iter().filter(|x| x % 2 == 0).collect();
```

### HashMap

```sigil
use std·collections·HashMap;

≔ map! = HashMap·new();
map.insert("key1", "value1");
map.insert("key2", "value2");

// Access
match map.get("key1") {
    Some(value) => println(value),
    None => println("not found"),
}

// Check existence
if map.contains_key("key1") {
    // ...
}

// Iteration
∀ (key, value) ∈ map {
    println(key + " => " + value);
}

// Remove
map.remove("key1");
```

### HashSet

```sigil
use std·collections·HashSet;

≔ set! = HashSet·new();
set.insert(1);
set.insert(2);
set.insert(1);  // No duplicates

≔ has_one = set.contains(&1);  // true
≔ len = set.len();              // 2
```

---

## std·string

### String Methods

```sigil
≔ s = "Hello, World!";

// Properties
≔ len = s.len();
≔ empty = s.is_empty();

// Transformations
≔ upper = s.to_uppercase();
≔ lower = s.to_lowercase();
≔ trimmed = s.trim();

// Search
≔ contains = s.contains("World");
≔ starts = s.starts_with("Hello");
≔ ends = s.ends_with("!");
≔ pos = s.find("World");  // Option<usize>

// Split and join
≔ parts = s.split(", ").collect();
≔ joined = parts.join(" - ");

// Replace
≔ replaced = s.replace("World", "Sigil");

// Substring
≔ sub = s[0..5];  // "Hello"
```

---

## std·http

### HTTP Client

```sigil
use std·http·{Client, Request};

≔ client = Client·new();

// GET request
≔ response = client.get("https://api.example.com/data")~?;
println("Status: " + str(response.status));
println("Body: " + response.body);

// GET with headers
≔ response = client
    .get("https://api.example.com/data")
    .header("Authorization", "Bearer token")
    .header("Accept", "application/json")
    .send()~?;

// POST request
≔ response = client
    .post("https://api.example.com/data")
    .header("Content-Type", "application/json")
    .body(r#"{"key": "value"}"#)
    .send()~?;

// Other methods
client.put(url)...
client.delete(url)...
client.patch(url)...
```

### Response

```sigil
Σ Response {
    status: i32,
    headers: HashMap<String, String>,
    body: String,
}
```

---

## std·json

### Parsing

```sigil
use std·json;

≔ text = r#"{"name": "Alice", "age": 30}"#;
≔ value = json·parse(text)~?;

// Access fields
≔ name = value["name"].as_str()~;
≔ age = value["age"].as_i32()~;
```

### Serialization

```sigil
use std·json;

≔ obj = json·object();
obj.set("name", "Alice");
obj.set("age", 30);

≔ text = json·stringify(&obj);
// {"name":"Alice","age":30}
```

---

## std·time

```sigil
use std·time·{Instant, Duration, DateTime};

// Timing
≔ start = Instant·now();
// ... do work ...
≔ elapsed = start.elapsed();
println("Took: " + str(elapsed.as_millis()) + "ms");

// Sleep
std·time·sleep(Duration·from_secs(1));

// Current time
≔ now = DateTime·now();
println(now.format("%Y-%m-%d %H:%M:%S"));
```

---

## std·mem

### Rc (Reference Counting)

```sigil
use std·mem·Rc;

≔ data = Rc·new(vec![1, 2, 3]);
≔ clone1 = data.clone();
≔ clone2 = data.clone();

println("Count: " + str(Rc·strong_count(&data)));  // 3
```

### Cell (Interior Mutability)

```sigil
use std·mem·Cell;

≔ cell = Cell·new(5);
≔ value = cell.get();
cell.set(10);
```

### Drop Trait

```sigil
trait Drop {
    λ drop(&Δ this);
}

Σ Resource {
    handle: i32,
}

⊢ Drop for Resource {
    λ drop(&Δ this) {
        // Cleanup when Resource goes out of scope
        println("Releasing resource " + str(this.handle));
    }
}
```

---

## std·async

### Async Functions

```sigil
use std·async;

async λ fetch_data(url: &str) → Result<String, Error> {
    ≔ response = http·get(url).await~?;
    Ok(response.body)
}

// Await multiple futures
≔ results = async·join_all([
    fetch_data("url1"),
    fetch_data("url2"),
    fetch_data("url3"),
]).await;
```

---

## Prelude

These are automatically imported into every Sigil program:

- `Option<T>` (Some, None)
- `Result<T, E>` (Ok, Err)
- `String`
- `Vec<T>`
- `print`, `println`, `eprintln`
- `str` (conversion function)
- `assert`, `assert_eq`

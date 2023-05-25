
# Fields
Every field element has type `FieldElement`. These are the concrete elements that can be added, multiplied and so on. The operations are delegated to a zero-sized struct implementing the trait `IsField` that defines the operation laws and algorithms used. There are already some basic fields defined.

We have a two layered model for fields. In one layer we have algorithms and optimizations at the field level, and on the other we have algorithms at the “big int” level. Both layers are interchangeable.
Our “Big Int” has the advantage that it’s size is fixed in compilation time through the use of generics. This removes the need for heap allocations and having logic to work with arbitrary sized numbers, and it’s not an issue because in proving system we always know the size of the fields in advance.
Over that we build different backends with different optimizations. The main backend we are using is Montgomery, to have a fast Montgomery multiplication, but it can be changed if for the use case another one is more useful.

Another important thing is that we're making sure that this is also in sync with webgpu, cuda and metal support

## How to instantiate a field element for a specific field
```rust
use math::field::element::FieldElement;
use math::field::fields::U64PrimeField;

let x = FieldElement::<U64PrimeField<11>>::from(3);
let y = FieldElement::<U64PrimeField<11>>::from(2);
let z = x + y;
```
Here x, y and z are elements of the field with order 11.

## How to work with general fields
```rust
use math::field::element::FieldElement;
use math::field::traits::IsField;

fn do_something_with_fe<F: IsField>(x: FieldElement<F>) {
    let y = x.pow(3)  + x.pow(2) + x;
    
    // ...
}
```
Here `F` is the Field and `x` is a field element. When further traits are needed for the field element you can specify them using the `where` notation:

```rust
use math::field::element::FieldElement;
use math::field::traits::IsField;

fn do_something_with_fe<F>(x: FieldElement<F>) 
    where
        F: IsField,
        FieldElement<F>: ByteConversion {
    let bytes = x.to_bytes_be();
    
}
```
## How to create a new Field
If you want to create your own field, for example to optimize an operation, you can do that by implementing the `IsField` trait for your own zero-sized struct:

```rust
struct MyCustomField;

impl IsField for MyCustomField {
    type BaseType = u128;

    fn add(a: &Self::BaseType, b: &Self::BaseType) -> Self::BaseType {
        // My optimized algorithm for adding
    }

    // ...
}
```

If we want to create elements of this new field, we can do so by:

```rust
fn main() {
    let x = FieldElement::<MyCustomField>::from(3);
    let y = x.pow(3) - x.pow(2) + x;
}
```

As you can see all the operators are already implemented automatically for you.

use lambdaworks_math::{
    field::{element::FieldElement, traits::IsFFTField},
    traits::ByteConversion,
};
use serde::{
    de::{MapAccess, SeqAccess, Visitor},
    ser::SerializeStruct,
    Deserialize, Serialize, Serializer,
};

/// A two-dimensional Table holding field elements, arranged in a row-major order.
/// This is the basic underlying data structure used for any two-dimensional component in the
/// the STARK protocol implementation, such as the `TraceTable` and the `EvaluationFrame`.
/// Since this struct is a representation of a two-dimensional table, all rows should have the same
/// length.
#[derive(Clone, Default, Debug, PartialEq, Eq)]
pub struct Table<F: IsFFTField> {
    pub data: Vec<FieldElement<F>>,
    pub width: usize,
    pub height: usize,
}

impl<F: IsFFTField> Table<F> {
    /// Crates a new Table instance from a one-dimensional array in row major order
    /// and the intended width of the table.
    pub fn new(data: &[FieldElement<F>], width: usize) -> Self {
        // Check if the intented width is 0, used for creating an empty table.
        if width == 0 {
            return Self {
                data: Vec::new(),
                width,
                height: 0,
            };
        }

        // Check that the one-dimensional data makes sense to be interpreted as a 2D one.
        assert!(crate::debug::validate_2d_structure(data, width));
        let height = data.len() / width;

        Self {
            data: data.to_vec(),
            width,
            height,
        }
    }

    /// Creates a Table instance from a vector of the intended columns.
    pub fn from_columns(columns: &[Vec<FieldElement<F>>]) -> Self {
        if columns.is_empty() {
            return Self::new(&Vec::new(), 0);
        }
        let height = columns[0].len();

        // Check that all columns have the same length for integrity
        debug_assert!(columns.iter().all(|c| c.len() == height));

        let width = columns.len();
        let mut data = Vec::with_capacity(width * height);
        for row_idx in 0..height {
            for column in columns {
                data.push(column[row_idx].clone());
            }
        }

        Self::new(&data, width)
    }

    /// Returns a vector of vectors of field elements representing the table rows
    pub fn rows(&self) -> Vec<Vec<FieldElement<F>>> {
        self.data.chunks(self.width).map(|r| r.to_vec()).collect()
    }

    /// Given a row index, returns a reference to that row as a slice of field elements.
    pub fn get_row(&self, row_idx: usize) -> &[FieldElement<F>] {
        let row_offset = row_idx * self.width;
        &self.data[row_offset..row_offset + self.width]
    }

    /// Given a row index, returns a mutable reference to that row as a slice of field elements.
    pub fn get_row_mut(&mut self, row_idx: usize) -> &mut [FieldElement<F>] {
        let n_cols = self.width;
        let row_offset = row_idx * n_cols;
        &mut self.data[row_offset..row_offset + n_cols]
    }

    /// Given a slice of field elements representing a row, appends it to
    /// the end of the table.
    pub fn append_row(&mut self, row: &[FieldElement<F>]) {
        debug_assert_eq!(row.len(), self.width);
        self.data.extend_from_slice(row);
        self.height += 1
    }

    /// Returns a reference to the last row of the table
    pub fn last_row(&self) -> &[FieldElement<F>] {
        self.get_row(self.height - 1)
    }

    /// Returns a vector of vectors of field elements representing the table
    /// columns
    pub fn columns(&self) -> Vec<Vec<FieldElement<F>>> {
        (0..self.width)
            .map(|col_idx| {
                (0..self.height)
                    .map(|row_idx| self.data[row_idx * self.width + col_idx].clone())
                    .collect()
            })
            .collect()
    }

    /// Given row and column indexes, returns the stored field element in that position of the table.
    pub fn get(&self, row: usize, col: usize) -> FieldElement<F> {
        let idx = row * self.width + col;
        self.data[idx].clone()
    }
}

// #[cfg(feature = "lambdaworks-serde")]
impl<F: IsFFTField> Serialize for Table<F>
where
    FieldElement<F>: ByteConversion,
{
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut state = serializer.serialize_struct("Table", 3)?;
        state.serialize_field("data", &self.data)?;
        state.serialize_field("width", &self.width)?;
        state.serialize_field("height", &self.height)?;
        state.end()
    }
}

// #[cfg(feature = "lambdaworks-serde")]
impl<'de, F: IsFFTField> Deserialize<'de> for Table<F>
where
    FieldElement<F>: ByteConversion,
{
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(Deserialize)]
        #[serde(field_identifier, rename_all = "lowercase")]
        enum Field {
            Data,
            Width,
            Height,
        }

        struct TableVisitor<F: IsFFTField>(std::marker::PhantomData<F>);

        impl<'de, F: IsFFTField> Visitor<'de> for TableVisitor<F> {
            type Value = Table<F>;

            fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
                formatter.write_str("struct Table")
            }

            fn visit_seq<V>(self, mut seq: V) -> Result<Table<F>, V::Error>
            where
                V: SeqAccess<'de>,
            {
                let data = seq
                    .next_element()?
                    .ok_or_else(|| serde::de::Error::invalid_length(0, &self))?;
                let width = seq
                    .next_element()?
                    .ok_or_else(|| serde::de::Error::invalid_length(1, &self))?;
                let height = seq
                    .next_element()?
                    .ok_or_else(|| serde::de::Error::invalid_length(2, &self))?;
                Ok(Table {
                    data,
                    width,
                    height,
                })
            }

            fn visit_map<V>(self, mut map: V) -> Result<Table<F>, V::Error>
            where
                V: MapAccess<'de>,
            {
                let mut data = None;
                let mut width = None;
                let mut height = None;

                while let Some(key) = map.next_key()? {
                    match key {
                        Field::Data => {
                            if data.is_some() {
                                return Err(serde::de::Error::duplicate_field("data"));
                            }
                            data = Some(map.next_value()?);
                        }
                        Field::Width => {
                            if width.is_some() {
                                return Err(serde::de::Error::duplicate_field("width"));
                            }
                            width = Some(map.next_value()?);
                        }
                        Field::Height => {
                            if height.is_some() {
                                return Err(serde::de::Error::duplicate_field("height"));
                            }
                            height = Some(map.next_value()?);
                        }
                    }
                }
                let data = data.ok_or_else(|| serde::de::Error::missing_field("data"))?;
                let width = width.ok_or_else(|| serde::de::Error::missing_field("width"))?;
                let height = height.ok_or_else(|| serde::de::Error::missing_field("height"))?;
                Ok(Table {
                    data,
                    width,
                    height,
                })
            }
        }

        const FIELDS: &'static [&'static str] = &["data", "width", "height"];
        deserializer.deserialize_struct("Table", FIELDS, TableVisitor(std::marker::PhantomData))
    }
}

use super::trace::TraceTable;
use crate::table::Table;
use lambdaworks_math::{
    field::{element::FieldElement, traits::IsFFTField},
    polynomial::Polynomial,
    traits::ByteConversion,
};
use serde::{
    de::{SeqAccess, Visitor},
    ser::SerializeStruct,
    Deserialize, Serialize, Serializer,
};

#[derive(Clone, Debug, PartialEq)]
pub struct Frame<F: IsFFTField> {
    table: Table<F>,
}

impl<F: IsFFTField> Frame<F> {
    pub fn new(data: Vec<FieldElement<F>>, row_width: usize) -> Self {
        let table = Table::new(&data, row_width);
        Self { table }
    }

    pub fn n_rows(&self) -> usize {
        self.table.height
    }

    pub fn n_cols(&self) -> usize {
        self.table.width
    }

    pub fn get_row(&self, row_idx: usize) -> &[FieldElement<F>] {
        self.table.get_row(row_idx)
    }

    pub fn get_row_mut(&mut self, row_idx: usize) -> &mut [FieldElement<F>] {
        self.table.get_row_mut(row_idx)
    }

    pub fn read_from_trace(
        trace: &TraceTable<F>,
        step: usize,
        blowup: u8,
        offsets: &[usize],
    ) -> Self {
        // Get trace length to apply module with it when getting elements of
        // the frame from the trace.
        let trace_steps = trace.n_rows();
        let data = offsets
            .iter()
            .flat_map(|frame_row_idx| {
                trace
                    .get_row((step + (frame_row_idx * blowup as usize)) % trace_steps)
                    .to_vec()
            })
            .collect();

        Self::new(data, trace.table.width)
    }

    /// Given a slice of trace polynomials, an evaluation point `x`, the frame offsets
    /// corresponding to the computation of the transitions, and a primitive root,
    /// outputs the trace evaluations of each trace polynomial over the values used to
    /// compute a transition.
    /// Example: For a simple Fibonacci computation, if t(x) is the trace polynomial of
    /// the computation, this will output evaluations t(x), t(g * x), t(g^2 * z).
    pub fn get_trace_evaluations(
        trace_polys: &[Polynomial<FieldElement<F>>],
        x: &FieldElement<F>,
        frame_offsets: &[usize],
        primitive_root: &FieldElement<F>,
    ) -> Vec<Vec<FieldElement<F>>> {
        frame_offsets
            .iter()
            .map(|offset| x * primitive_root.pow(*offset))
            .map(|eval_point| {
                trace_polys
                    .iter()
                    .map(|poly| poly.evaluate(&eval_point))
                    .collect::<Vec<FieldElement<F>>>()
            })
            .collect()
    }
}

impl<F: IsFFTField> Serialize for Frame<F>
where
    FieldElement<F>: ByteConversion,
{
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut state = serializer.serialize_struct("Frame", 1)?;
        state.serialize_field("table", &self.table)?;
        state.end()
    }
}

impl<'de, F: IsFFTField> Deserialize<'de> for Frame<F> {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        // Declare fields of the struct
        #[derive(Deserialize)]
        #[serde(field_identifier, rename_all = "lowercase")]
        enum Field {
            Table,
        }

        // Visitor of struct to deserialize
        struct FrameVisitor<F: IsFFTField>(std::marker::PhantomData<F>);

        impl<'de, F: IsFFTField> Visitor<'de> for FrameVisitor<F> {
            type Value = Frame<F>;

            fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
                formatter.write_str("struct Frame")
            }

            fn visit_seq<V>(self, mut seq: V) -> Result<Frame<F>, V::Error>
            where
                V: SeqAccess<'de>,
            {
                let table = seq
                    .next_element()?
                    .ok_or_else(|| serde::de::Error::invalid_length(0, &self))?;
                Ok(Frame { table })
            }

            fn visit_map<V>(self, mut map: V) -> Result<Frame<F>, V::Error>
            where
                V: serde::de::MapAccess<'de>,
            {
                let mut table = None;
                while let Some(key) = map.next_key()? {
                    match key {
                        Field::Table => {
                            if table.is_some() {
                                return Err(serde::de::Error::duplicate_field("table"));
                            }
                            table = Some(map.next_value()?);
                        }
                    }
                }
                let table = table.ok_or_else(|| serde::de::Error::missing_field("table"))?;
                Ok(Frame { table })
            }
        }

        const FIELDS: &'static [&'static str] = &["table"];
        deserializer.deserialize_struct("Frame", FIELDS, FrameVisitor(std::marker::PhantomData))
    }
}

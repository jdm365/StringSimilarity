use std::cmp;
use pyo3::prelude::*;
use pyo3::types::{ PyString, PyList };

use rayon::prelude::*;


#[pyfunction]
#[pyo3(signature = (str1, str2, max_prefix_length=4, scaling_factor=0.1))]
fn jaro_winkler_similarity(
    _py: Python, 
    str1: Option<&PyString>, 
    str2: Option<&PyString>,
    max_prefix_length: Option<i32>,
    scaling_factor: Option<f32>,
    ) -> PyResult<f32> {

    if let (Some(str1), Some(str2)) = (str1, str2) {
        let str1_bytes = str1.to_str()?.as_bytes();
        let str2_bytes = str2.to_str()?.as_bytes();

        Ok(get_approx_jaro_winkler_similarity(
            str1_bytes,
            str2_bytes,
            max_prefix_length.unwrap_or(4) as usize,
            scaling_factor.unwrap_or(0.1)
        ))
    } else {
        Ok(0.0)
    }
    /*
    // if str1 or str2 is None, return 0
    if str1.is_none() || str2.is_none() {
        return Ok(0.0);
    }

    /*
    return Ok(get_jaro_winkler_similarity(
            &str1.expect("str1 is None").extract::<String>().unwrap().into_bytes().to_vec(),
            &str2.expect("str2 is None").extract::<String>().unwrap().into_bytes().to_vec(),
            max_prefix_length.unwrap_or(4) as usize,
            scaling_factor.unwrap_or(0.1)
            ));
    */
    return Ok(get_approx_jaro_winkler_similarity(
            &str1.unwrap().to_string().as_bytes().to_vec(),
            &str2.unwrap().to_string().as_bytes().to_vec(),
            max_prefix_length.unwrap_or(4) as usize,
            scaling_factor.unwrap_or(0.1)
            ));
    */
}

#[pyfunction]
fn jaro_winkler_similarity_batched(
    py: Python, 
    str1_list: &PyList, 
    str2_list: &PyList,
    max_prefix_length: Option<i32>,
    scaling_factor: Option<f32>,
) -> PyResult<Vec<f32>> {

    let str1_bytes: Vec<Vec<u8>> = str1_list.iter()
        .map(|py_string| py_string.extract::<String>().unwrap().into_bytes())
        .collect();
    let str2_bytes: Vec<Vec<u8>> = str2_list.iter()
        .map(|py_string| py_string.extract::<String>().unwrap().into_bytes())
        .collect();

    if str1_bytes.len() != str2_bytes.len() {
        return Err(pyo3::exceptions::PyValueError::new_err("str1_list and str2_list must be of the same length"));
    }

    // Use py.allow_threads() to release the GIL
    let jw_sims = py.allow_threads(|| {
        str1_bytes.par_iter().zip(str2_bytes.par_iter()).map(|(str1, str2)| {
            get_approx_jaro_winkler_similarity(
                str1.as_slice(),
                str2.as_slice(),
                max_prefix_length.unwrap_or(4) as usize,
                scaling_factor.unwrap_or(0.1),
            )
        }).collect()
    });

    Ok(jw_sims)
}


#[pyfunction]
#[pyo3(signature = (str1, str2, deletion_cost=1, insertion_cost=1, substitution_cost=1))]
fn weighted_levenshtein_distance(
    _py: Python, 
    str1: Option<&PyString>, 
    str2: Option<&PyString>,
    deletion_cost: Option<i32>,
    insertion_cost: Option<i32>,
    substitution_cost: Option<i32>,
    ) -> PyResult<usize> {

    // if str1 or str2 is None, return 0
    if str1.is_none() || str2.is_none() {
        return Ok(0);
    }

    Ok(get_weighted_levenshtein_distance(
            &str1.unwrap().to_string().as_bytes().to_vec(),
            &str2.unwrap().to_string().as_bytes().to_vec(),
            deletion_cost.unwrap_or(1) as usize,
            insertion_cost.unwrap_or(1) as usize,
            substitution_cost.unwrap_or(1) as usize,
            )) 
}


#[pymodule]
fn string_sim_metrics(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(jaro_winkler_similarity, m)?)?;
    m.add_function(wrap_pyfunction!(weighted_levenshtein_distance, m)?)?;
    m.add_function(wrap_pyfunction!(jaccard_similarity, m)?)?;
    m.add_function(wrap_pyfunction!(jaro_winkler_similarity_batched, m)?)?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add("__author__", env!("CARGO_PKG_AUTHORS"))?;
    m.add("__description__", env!("CARGO_PKG_DESCRIPTION"))?;
    Ok(())
}

#[inline]
pub fn get_approx_jaro_winkler_similarity(
    str1: &[u8],
    str2: &[u8],
    max_prefix_length: usize, 
    scaling_factor: f32
    ) -> f32 {
    let len1 = str1.len();
    let len2 = str2.len();

    if len1 == 0 || len2 == 0 {
        return 0.0;
    }

    if str1 == str2 {
        return 1.0;
    }

    let search_range = cmp::max(len1, len2) / 2 - 1;

    // Bitwise representation of matches
    let mut matches1 = 0u64;
    let mut matches2 = 0u64;

    // Find matches using bitwise operations
    for (idx, &char1) in str1.iter().enumerate() {
        let start = idx.saturating_sub(search_range);
        let end = cmp::min(idx + search_range + 1, len2);

        for (jdx, &char2) in str2.iter().enumerate().take(end).skip(start) {
            if char1 == char2 && (matches2 & (1 << jdx)) == 0 {
                matches1 |= 1 << idx;
                matches2 |= 1 << jdx;
                break;
            }
        }
    }

    let matched_count = matches1.count_ones();
    if matched_count == 0 {
        return 0.0;
    }

    // Count transpositions
    let mut transpositions = 0;
    let mut k = 0;
    for (idx, &char1) in str1.iter().enumerate() {
        if (matches1 & (1 << idx)) != 0 {
            while (matches2 & (1 << k)) == 0 { k += 1; }
            if char1 != str2[k] { transpositions += 1; }
            k += 1;
        }
    }

    let jaro = (
        (matched_count as f32) / (len1 as f32) +
        (matched_count as f32) / (len2 as f32) +
        ((matched_count - transpositions / 2) as f32) / (matched_count as f32)
    ) / 3.0;

    // Calculate Jaro-Winkler
    let common_prefix = str1.iter().zip(str2).take(max_prefix_length).filter(|&(c1, c2)| c1 == c2).count();
    return jaro + (common_prefix as f32 * scaling_factor * (1.0 - jaro));
}

#[inline]
pub fn get_jaro_winkler_similarity(
    str1: &Vec<u8>, 
    str2: &Vec<u8>,
    max_prefix_length: usize,
    scaling_factor: f32,
    ) -> f32 {
    let len1 = str1.len();
    let len2 = str2.len();

    if len1 == 0 || len2 == 0 {
        return 0.0;
    }

    if str1 == str2 {
        return 1.0;
    }

    let search_range = (cmp::max(len1, len2) / 2) - 1;

    let mut n_matches = 0;

    // let mut hash1: Vec<u8> = vec![0; len1 + (len1 % 32)];
    // let mut hash2: Vec<u8> = vec![0; len2 + (len2 % 32)];
    let mut hash1: Vec<u8> = vec![0; len1];
    let mut hash2: Vec<u8> = vec![0; len2];

    for idx in 0..len1 {
        let start = cmp::max(0, idx as i32 - search_range as i32) as usize;
        let end   = cmp::min(len2, idx + search_range + 1);

        for jdx in start..end {
            if (str1[idx] == str2[jdx]) && (hash2[jdx] == 0) {
                hash1[idx] = 1;
                hash2[jdx] = 1;
                n_matches += 1;
                break;
            }
        }
    }


    if n_matches == 0 {
        return 0.0;
    }

    let mut n_transpositions: f32 = 0.0;
    let mut i = 0;

    for idx in 0..len1 {
        if hash1[idx] == 1 {
            while hash2[i] == 0 {
                i += 1;
            }
            if str1[idx] != str2[i] {
                n_transpositions += 1.0;
            }
            i += 1;
        }
    }
    n_transpositions = (0.5 * n_transpositions.floor()).floor();


    let sim = ((n_matches as f32 / (len1 as f32)) + (n_matches as f32 / (len2 as f32)) + ((n_matches as f32 - n_transpositions) / n_matches as f32)) / 3.0;

    // Now get convert jaro to jaro_winkler_similarity
    let mut prefix = 0.0;
    for idx in 0..cmp::min(max_prefix_length, cmp::min(len1, len2)) {
        if str1[idx] == str2[idx] {
            prefix += 1.0;
        } else {
            break;
        }
    }
    return sim + (prefix * scaling_factor * (1.0 - sim));
}


pub fn get_weighted_levenshtein_distance(
    str1: &Vec<u8>, 
    str2: &Vec<u8>,
    deletion_cost: usize,
    insertion_cost: usize,
    substitution_cost: usize
    ) -> usize {
    let len1 = str1.len();
    let len2 = str2.len();

    if len1 == 0 || len2 == 0 {
        return 0;
    }

    if str1 == str2 {
        return 1;
    }

    let mut table: Vec<Vec<usize>> = vec![vec![0; len2 + 1]; len1 + 1];

    for idx in 1..(len1 + 1) {
        table[idx][0] = table[idx - 1][0] + deletion_cost;
    }

    for idx in 1..(len2 + 1) {
        table[0][idx] = table[0][idx - 1] + insertion_cost;
    }

    for (idx, c1) in str1.iter().enumerate() {
        for (jdx, c2) in str2.iter().enumerate() {
            let sub_cost = if c1 == c2 { 0 } else { substitution_cost };
            table[idx + 1][jdx + 1] = cmp::min(
                cmp::min(
                    table[idx][jdx + 1] + deletion_cost, 
                    table[idx + 1][jdx] + insertion_cost
                    ), 
                table[idx][jdx] + sub_cost
                );
        }
    }
    return table[len1][len2];
}

fn get_jaccard_similarity(str1: &[u8], str2: &[u8]) -> f32 {
    const ASCII_CHAR_COUNT: usize = 128;
    let mut char_presence1 = [false; ASCII_CHAR_COUNT];
    let mut char_presence2 = [false; ASCII_CHAR_COUNT];

    // Iterate through the bytes of str1 and str2
    for &byte in str1 {
        if byte < ASCII_CHAR_COUNT as u8 {
            char_presence1[byte as usize] = true;
        }
    }

    for &byte in str2 {
        if byte < ASCII_CHAR_COUNT as u8 {
            char_presence2[byte as usize] = true;
        }
    }

    let mut intersection_count = 0;
    let mut union_count = 0;

    for i in 0..ASCII_CHAR_COUNT {
        let in_str1 = char_presence1[i];
        let in_str2 = char_presence2[i];

        if in_str1 || in_str2 {
            union_count += 1;
            if in_str1 && in_str2 {
                intersection_count += 1;
            }
        }
    }

    intersection_count as f32 / union_count as f32
}

#[pyfunction]
fn jaccard_similarity(
    _py: Python, 
    str1: Option<&PyString>, 
    str2: Option<&PyString>,
) -> PyResult<f32> {
    const MAX_STR_LEN: usize = 256; // Adjust this based on your expected max string length

    match (str1, str2) {
        (Some(str1), Some(str2)) => {
            let mut buffer1 = [0u8; MAX_STR_LEN];
            let mut buffer2 = [0u8; MAX_STR_LEN];

            // Copy characters into buffers, ensuring we don't exceed their lengths
            let len1 = str1.to_string().bytes().take(MAX_STR_LEN).enumerate().map(|(i, b)| { buffer1[i] = b; i + 1 }).last().unwrap_or(0);
            let len2 = str2.to_string().bytes().take(MAX_STR_LEN).enumerate().map(|(i, b)| { buffer2[i] = b; i + 1 }).last().unwrap_or(0);

            // Slices of the actual lengths
            let slice1 = &buffer1[..len1];
            let slice2 = &buffer2[..len2];

            // Call the modified Jaccard function with slices
            Ok(get_jaccard_similarity(slice1, slice2))
        },
        _ => Ok(0.0),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_error() {
        let str1: Vec<u8> = "testdklfj;asdkljfakl;jsdlk;fjasklj;df".to_string().as_bytes().to_vec();
        let str2: Vec<u8> = "tasdklfaskl;djfjas;lkjdfkl;jasdest".to_string().as_bytes().to_vec();

        let similarity = get_jaro_winkler_similarity(&str1, &str2, 1, 0.1);
        let _similarity_wlev = get_weighted_levenshtein_distance(&str1, &str2, 1, 1, 1);

        assert!(similarity <= 1.0);
        assert!(similarity >= 0.0);
    }

}

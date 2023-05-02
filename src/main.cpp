/**
 ********************************************************************************
 * @file    main.cpp
 * @author  [abdeladim-s](https://github.com/abdeladim-s)
 * @date    2023
 * @brief   Python bindings for GPT-J Language model based on [ggml](https://github.com/ggerganov/ggml)
 * @par     ggml is licensed under MIT Copyright (c) 2022 Georgi Gerganov,
            please see [ggml License](./GGML_LICENSE)
 ********************************************************************************
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <map>

//PYBIND11_MAKE_OPAQUE(std::vector<float>);

#include "utils.h"
#include "gptj.h"
#include "main.h"



#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)


namespace py = pybind11;
using namespace pybind11::literals; // to bring in the `_a` literal

gpt_vocab::id gpt_sample_top_k_top_p_wrapper(
        const gpt_vocab & vocab,
        py::array_t<float> logits,
        int    top_k,
        double top_p,
        double temp,
        int seed){

    py::buffer_info buf1 = logits.request();
    auto *logits_ptr = static_cast<float *>(buf1.ptr);
    std::mt19937 rng(seed);

    return gpt_sample_top_k_top_p(vocab, logits_ptr, top_k, top_p, temp, rng);
}

py::tuple gptj_eval_wrapper(
        const gptj_model & model,
        const int n_threads,
        const int n_past,
        const std::vector<gpt_vocab::id> & embd_inp,
        size_t mem_per_token){

    std::vector<float> embd_w;
    size_t mpt = mem_per_token;
    auto res = gptj_eval(model, n_threads, n_past, embd_inp, embd_w, mpt);

    py::tuple tup = py::make_tuple(embd_w, mpt);
    return tup;
}



PYBIND11_MODULE(_pygptj, m) {
    m.doc() = R"pbdoc(
        PyGPT-J: Python binding to GPT-J
        -----------------------

        .. currentmodule:: _pygptj

        .. autosummary::
           :toctree: _generate

    )pbdoc";

    py::class_<gptj_gpt_params>(m,"gptj_gpt_params" /*,py::dynamic_attr()*/)
        .def(py::init<>())
        .def_readwrite("seed", &gptj_gpt_params::seed)
        .def_readwrite("n_threads", &gptj_gpt_params::n_threads)
        .def_readwrite("n_predict", &gptj_gpt_params::n_predict)
        .def_readwrite("top_k", &gptj_gpt_params::top_k)
        .def_readwrite("top_p", &gptj_gpt_params::top_p)
        .def_readwrite("temp", &gptj_gpt_params::temp)
        .def_readwrite("n_batch", &gptj_gpt_params::n_batch)
        .def_readwrite("model", &gptj_gpt_params::model)
        .def_readwrite("prompt", &gptj_gpt_params::prompt)
        ;

    py::class_<gptj_hparams>(m,"gptj_hparams" /*,py::dynamic_attr()*/)
        .def(py::init<>())
        .def_readwrite("n_vocab", &gptj_hparams::n_vocab)
        .def_readwrite("n_ctx", &gptj_hparams::n_ctx)
        .def_readwrite("n_embd", &gptj_hparams::n_embd)
        .def_readwrite("n_head", &gptj_hparams::n_head)
        .def_readwrite("n_layer", &gptj_hparams::n_layer)
        .def_readwrite("n_rot", &gptj_hparams::n_rot)
        .def_readwrite("f16", &gptj_hparams::f16)
        ;
    py::class_<gptj_model>(m,"gptj_model" /*,py::dynamic_attr()*/)
        .def(py::init<>())
    ;

 py::class_<gpt_vocab>(m,"gpt_vocab" /*,py::dynamic_attr()*/)
        .def(py::init<>())
        .def_readwrite("token_to_id", &gpt_vocab::token_to_id)
        .def_readwrite("id_to_token", &gpt_vocab::id_to_token)

    ;

py::class_<gptj_context>(m,"gptj_context" /*,py::dynamic_attr()*/)
;

    m.def("gptj_model_load", &gptj_model_load);

    m.def("gptj_eval", &gptj_eval_wrapper);
    m.def("gptj_free", &gptj_free);
    m.def("gpt_sample_top_k_top_p", &gpt_sample_top_k_top_p_wrapper);
    m.def("gpt_tokenize", &gpt_tokenize);
    m.def("gpt_vocab_init", &gpt_vocab_init);

    m.def("gptj_generate", &gptj_generate);




#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif
}

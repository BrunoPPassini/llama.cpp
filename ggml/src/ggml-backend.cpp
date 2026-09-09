// Note: porting this file to C++ is a work in progress

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#   define NOMINMAX
#endif
#include <windows.h>
#endif

#include "ggml-backend.h"
#include "ggml-backend-impl.h"
#include "ggml-alloc.h"
#include "ggml-impl.h"

#include <assert.h>
#include <limits.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <algorithm>
#include <condition_variable>
#include <mutex>
#include <thread>
#include <vector>

#ifdef __APPLE__
#include <sys/types.h>
#include <sys/sysctl.h>
#endif


// backend buffer type

const char * ggml_backend_buft_name(ggml_backend_buffer_type_t buft) {
    GGML_ASSERT(buft);
    return buft->iface.get_name(buft);
}

ggml_backend_buffer_t ggml_backend_buft_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    GGML_ASSERT(buft);
    if (size == 0) {
        // return a dummy buffer for zero-sized allocations
        return ggml_backend_buffer_init(buft, {}, NULL, 0);
    }
    return buft->iface.alloc_buffer(buft, size);
}

size_t ggml_backend_buft_get_alignment(ggml_backend_buffer_type_t buft) {
    GGML_ASSERT(buft);
    return buft->iface.get_alignment(buft);
}

size_t ggml_backend_buft_get_max_size(ggml_backend_buffer_type_t buft) {
    GGML_ASSERT(buft);
    // get_max_size is optional, defaults to SIZE_MAX
    if (buft->iface.get_max_size) {
        return buft->iface.get_max_size(buft);
    }
    return SIZE_MAX;
}

size_t ggml_backend_buft_get_alloc_size(ggml_backend_buffer_type_t buft, const struct ggml_tensor * tensor) {
    GGML_ASSERT(buft);
    // get_alloc_size is optional, defaults to ggml_nbytes
    if (buft->iface.get_alloc_size) {
        size_t size = buft->iface.get_alloc_size(buft, tensor);
        assert(size >= ggml_nbytes(tensor));
        return size;
    }
    return ggml_nbytes(tensor);
}

bool ggml_backend_buft_is_host(ggml_backend_buffer_type_t buft) {
    GGML_ASSERT(buft);
    if (buft->iface.is_host) {
        return buft->iface.is_host(buft);
    }
    return false;
}

ggml_backend_dev_t ggml_backend_buft_get_device(ggml_backend_buffer_type_t buft) {
    GGML_ASSERT(buft);
    return buft->device;
}

// backend buffer

ggml_backend_buffer_t ggml_backend_buffer_init(
               ggml_backend_buffer_type_t buft,
        struct ggml_backend_buffer_i      iface,
               void *                     context,
               size_t                     size) {
    ggml_backend_buffer_t buffer = new ggml_backend_buffer {
        /* .interface = */ iface,
        /* .buft      = */ buft,
        /* .context   = */ context,
        /* .size      = */ size,
        /* .usage     = */ GGML_BACKEND_BUFFER_USAGE_ANY
    };

    return buffer;
}

const char * ggml_backend_buffer_name(ggml_backend_buffer_t buffer) {
    return ggml_backend_buft_name(ggml_backend_buffer_get_type(buffer));
}

void ggml_backend_buffer_free(ggml_backend_buffer_t buffer) {
    if (buffer == NULL) {
        return;
    }

    if (buffer->iface.free_buffer != NULL) {
        buffer->iface.free_buffer(buffer);
    }
    delete buffer;
}

size_t ggml_backend_buffer_get_size(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(buffer);
    return buffer->size;
}

void * ggml_backend_buffer_get_base(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(buffer);
    // get_base is optional if the buffer is zero-sized
    if (!ggml_backend_buffer_is_meta(buffer) && buffer->size == 0) {
        return NULL;
    }

    // FIXME JG: a multi_buffer has a non-zero size, according to the above comment get_base is not optional,
    //     I don't know whether the above comment is correct
    if (!buffer->iface.get_base) {
        return NULL;
    }

    void * base = buffer->iface.get_base(buffer);

    GGML_ASSERT(base != NULL && "backend buffer base cannot be NULL");

    return base;
}

enum ggml_status ggml_backend_buffer_init_tensor(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor) {
    GGML_ASSERT(buffer);
    // init_tensor is optional
    if (buffer->iface.init_tensor) {
        return buffer->iface.init_tensor(buffer, tensor);
    }
    return GGML_STATUS_SUCCESS;
}

void ggml_backend_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    GGML_ASSERT(buffer);
    // clear is optional if the buffer is zero-sized
    if (buffer->size == 0) {
        return;
    }

    buffer->iface.clear(buffer, value);
}

size_t ggml_backend_buffer_get_alignment(ggml_backend_buffer_t buffer) {
    return ggml_backend_buft_get_alignment(ggml_backend_buffer_get_type(buffer));
}

size_t ggml_backend_buffer_get_max_size(ggml_backend_buffer_t buffer) {
    return ggml_backend_buft_get_max_size(ggml_backend_buffer_get_type(buffer));
}

size_t ggml_backend_buffer_get_alloc_size(ggml_backend_buffer_t buffer, const struct ggml_tensor * tensor) {
    return ggml_backend_buft_get_alloc_size(ggml_backend_buffer_get_type(buffer), tensor);
}

bool ggml_backend_buffer_is_host(ggml_backend_buffer_t buffer) {
    return ggml_backend_buft_is_host(ggml_backend_buffer_get_type(buffer));
}

void ggml_backend_buffer_set_usage(ggml_backend_buffer_t buffer, enum ggml_backend_buffer_usage usage) {
    GGML_ASSERT(buffer);
    buffer->usage = usage;

    // FIXME: add a generic callback to the buffer interface
    if (ggml_backend_buffer_is_multi_buffer(buffer)) {
        ggml_backend_multi_buffer_set_usage(buffer, usage);
    }
}

enum ggml_backend_buffer_usage ggml_backend_buffer_get_usage(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(buffer);
    return buffer->usage;
}

ggml_backend_buffer_type_t ggml_backend_buffer_get_type(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(buffer);
    return buffer->buft;
}

void ggml_backend_buffer_reset(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(buffer);
    if (buffer->iface.reset) {
        buffer->iface.reset(buffer);
    }
}

bool ggml_backend_buffer_copy_tensor(const struct ggml_tensor * src, struct ggml_tensor * dst) {
    ggml_backend_buffer_t dst_buf = dst->view_src ? dst->view_src->buffer : dst->buffer;
    if (dst_buf->iface.cpy_tensor) {
        return dst_buf->iface.cpy_tensor(dst_buf, src, dst);
    }
    return false;
}

// backend

ggml_guid_t ggml_backend_guid(ggml_backend_t backend) {
    if (backend == NULL) {
        return NULL;
    }
    return backend->guid;
}

const char * ggml_backend_name(ggml_backend_t backend) {
    if (backend == NULL) {
        return "NULL";
    }
    return backend->iface.get_name(backend);
}

void ggml_backend_free(ggml_backend_t backend) {
    if (backend == NULL) {
        return;
    }

    backend->iface.free(backend);
}

ggml_backend_buffer_type_t ggml_backend_get_default_buffer_type(ggml_backend_t backend) {
    GGML_ASSERT(backend);
    return ggml_backend_dev_buffer_type(backend->device);
}

ggml_backend_buffer_t ggml_backend_alloc_buffer(ggml_backend_t backend, size_t size) {
    return ggml_backend_buft_alloc_buffer(ggml_backend_get_default_buffer_type(backend), size);
}

size_t ggml_backend_get_alignment(ggml_backend_t backend) {
    return ggml_backend_buft_get_alignment(ggml_backend_get_default_buffer_type(backend));
}

size_t ggml_backend_get_max_size(ggml_backend_t backend) {
    return ggml_backend_buft_get_max_size(ggml_backend_get_default_buffer_type(backend));
}

void ggml_backend_tensor_set_async(ggml_backend_t backend, struct ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    GGML_ASSERT(backend);
    GGML_ASSERT(tensor);
    GGML_ASSERT(tensor->data != NULL && "tensor not allocated");
    GGML_ASSERT(offset + size <= ggml_nbytes(tensor) && "tensor write out of bounds");

    if (backend->iface.set_tensor_async == NULL) {
        ggml_backend_synchronize(backend);
        ggml_backend_tensor_set(tensor, data, offset, size);
    } else {
        backend->iface.set_tensor_async(backend, tensor, data, offset, size);
    }
}

void ggml_backend_tensor_get_async(ggml_backend_t backend, const struct ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    GGML_ASSERT(backend);
    GGML_ASSERT(tensor);
    GGML_ASSERT(tensor->data != NULL && "tensor not allocated");
    GGML_ASSERT(offset + size <= ggml_nbytes(tensor) && "tensor read out of bounds");

    if (backend->iface.get_tensor_async == NULL) {
        ggml_backend_synchronize(backend);
        ggml_backend_tensor_get(tensor, data, offset, size);
    } else {
        backend->iface.get_tensor_async(backend, tensor, data, offset, size);
    }
}

void ggml_backend_tensor_set_2d_async(ggml_backend_t backend, struct ggml_tensor * tensor, const void * data, size_t offset, size_t size,
            size_t n_copies, size_t stride_tensor, size_t stride_data) {
    GGML_ASSERT(backend);
    GGML_ASSERT(tensor);
    GGML_ASSERT(tensor->data != NULL && "tensor not allocated");

    if (n_copies <= 1 || backend->iface.set_tensor_2d_async == NULL) {
        for (size_t i = 0; i < n_copies; i++) {
            ggml_backend_tensor_set_async(backend, tensor, (const char *) data + i*stride_data, offset + i*stride_tensor, size);
        }
        return;
    }
    if (size == 0) {
        return;
    }

    GGML_ASSERT(tensor->data != NULL && "tensor not allocated");
    GGML_ASSERT(offset + (n_copies-1)*stride_tensor + size <= ggml_nbytes(tensor) && "tensor write out of bounds");
    backend->iface.set_tensor_2d_async(backend, tensor, data, offset, size, n_copies, stride_tensor, stride_data);
}

void ggml_backend_tensor_get_2d_async(ggml_backend_t backend, const struct ggml_tensor * tensor, void * data, size_t offset, size_t size,
            size_t n_copies, size_t stride_tensor, size_t stride_data) {
    GGML_ASSERT(backend);
    GGML_ASSERT(tensor);
    GGML_ASSERT(tensor->data != NULL && "tensor not allocated");

    if (n_copies <= 1 || backend->iface.get_tensor_2d_async == NULL) {
        for (size_t i = 0; i < n_copies; i++) {
            ggml_backend_tensor_get_async(backend, tensor, (char *) data + i*stride_data, offset + i*stride_tensor, size);
        }
        return;
    }
    if (size == 0) {
        return;
    }

    GGML_ASSERT(tensor->data != NULL && "tensor not allocated");
    GGML_ASSERT(offset + (n_copies-1)*stride_tensor + size <= ggml_nbytes(tensor) && "tensor read out of bounds");
    backend->iface.get_tensor_2d_async(backend, tensor, data, offset, size, n_copies, stride_tensor, stride_data);
}

void ggml_backend_tensor_set(struct ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    GGML_ASSERT(tensor);
    ggml_backend_buffer_t buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;
    GGML_ASSERT(buf != NULL && "tensor buffer not set");

    if (size == 0) {
        return;
    }

    GGML_ASSERT(tensor->data != NULL && "tensor not allocated");
    GGML_ASSERT(offset + size <= ggml_nbytes(tensor) && "tensor write out of bounds");

    buf->iface.set_tensor(buf, tensor, data, offset, size);
}

void ggml_backend_tensor_get(const struct ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    GGML_ASSERT(tensor);
    ggml_backend_buffer_t buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;
    GGML_ASSERT(buf != NULL && "tensor buffer not set");

    if (size == 0) {
        return;
    }

    GGML_ASSERT(tensor->data != NULL && "tensor not allocated");
    GGML_ASSERT(offset + size <= ggml_nbytes(tensor) && "tensor read out of bounds");

    buf->iface.get_tensor(buf, tensor, data, offset, size);
}

void ggml_backend_tensor_set_2d(struct ggml_tensor * tensor, const void * data, size_t offset, size_t size,
            size_t n_copies, size_t stride_tensor, size_t stride_data) {
    GGML_ASSERT(tensor);
    ggml_backend_buffer_t buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;
    GGML_ASSERT(buf != NULL && "tensor buffer not set");

    if (n_copies <= 1 || buf->iface.set_tensor_2d == NULL) {
        for (size_t i = 0; i < n_copies; i++) {
            ggml_backend_tensor_set(tensor, (const char *) data + i*stride_data, offset + i*stride_tensor, size);
        }
        return;
    }
    if (size == 0) {
        return;
    }

    GGML_ASSERT(tensor->data != NULL && "tensor not allocated");
    GGML_ASSERT(offset + (n_copies-1)*stride_tensor + size <= ggml_nbytes(tensor) && "tensor write out of bounds");

    buf->iface.set_tensor_2d(buf, tensor, data, offset, size, n_copies, stride_tensor, stride_data);
}

void ggml_backend_tensor_get_2d(const struct ggml_tensor * tensor, void * data, size_t offset, size_t size,
            size_t n_copies, size_t stride_tensor, size_t stride_data) {
    GGML_ASSERT(tensor);
    ggml_backend_buffer_t buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;
    GGML_ASSERT(buf != NULL && "tensor buffer not set");

    if (n_copies <= 1 || buf->iface.get_tensor_2d == NULL) {
        for (size_t i = 0; i < n_copies; i++) {
            ggml_backend_tensor_get(tensor, (char *) data + i*stride_data, offset + i*stride_tensor, size);
        }
        return;
    }
    if (size == 0) {
        return;
    }

    GGML_ASSERT(tensor->data != NULL && "tensor not allocated");
    GGML_ASSERT(offset + (n_copies-1)*stride_tensor + size <= ggml_nbytes(tensor) && "tensor read out of bounds");

    buf->iface.get_tensor_2d(buf, tensor, data, offset, size, n_copies, stride_tensor, stride_data);
}

void ggml_backend_tensor_memset(struct ggml_tensor * tensor, uint8_t value, size_t offset, size_t size) {
    GGML_ASSERT(tensor);
    ggml_backend_buffer_t buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;

    if (size == 0) {
        return;
    }

    GGML_ASSERT(buf != NULL && "tensor buffer not set");
    GGML_ASSERT(tensor->data != NULL && "tensor not allocated");
    GGML_ASSERT(offset + size <= ggml_nbytes(tensor) && "tensor write out of bounds");
    GGML_ASSERT(buf->iface.memset_tensor != NULL && "memset not implemented by backend buffer");

    buf->iface.memset_tensor(buf, tensor, value, offset, size);
}

void ggml_backend_synchronize(ggml_backend_t backend) {
    GGML_ASSERT(backend);
    if (backend->iface.synchronize == NULL) {
        return;
    }

    backend->iface.synchronize(backend);
}

ggml_backend_graph_plan_t ggml_backend_graph_plan_create(ggml_backend_t backend, struct ggml_cgraph * cgraph) {
    GGML_ASSERT(backend);
    GGML_ASSERT(backend->iface.graph_plan_create != NULL);

    return backend->iface.graph_plan_create(backend, cgraph);
}

void ggml_backend_graph_plan_free(ggml_backend_t backend, ggml_backend_graph_plan_t plan) {
    GGML_ASSERT(backend);
    GGML_ASSERT(backend->iface.graph_plan_free != NULL);

    backend->iface.graph_plan_free(backend, plan);
}

enum ggml_status ggml_backend_graph_plan_compute(ggml_backend_t backend, ggml_backend_graph_plan_t plan) {
    GGML_ASSERT(backend);
    GGML_ASSERT(backend->iface.graph_plan_compute != NULL);

    return backend->iface.graph_plan_compute(backend, plan);
}

enum ggml_status ggml_backend_graph_compute(ggml_backend_t backend, struct ggml_cgraph * cgraph) {
    enum ggml_status err = ggml_backend_graph_compute_async(backend, cgraph);
    ggml_backend_synchronize(backend);
    return err;
}

enum ggml_status ggml_backend_graph_compute_async(ggml_backend_t backend, struct ggml_cgraph * cgraph) {
    GGML_ASSERT(backend);
    return backend->iface.graph_compute(backend, cgraph);
}

bool ggml_backend_supports_op(ggml_backend_t backend, const struct ggml_tensor * op) {
    GGML_ASSERT(backend);
    return ggml_backend_dev_supports_op(backend->device, op);
}

bool ggml_backend_supports_buft(ggml_backend_t backend, ggml_backend_buffer_type_t buft) {
    GGML_ASSERT(backend);
    return ggml_backend_dev_supports_buft(backend->device, buft);
}

bool ggml_backend_offload_op(ggml_backend_t backend, const struct ggml_tensor * op) {
    GGML_ASSERT(backend);
    return ggml_backend_dev_offload_op(backend->device, op);
}

ggml_backend_dev_t ggml_backend_get_device(ggml_backend_t backend) {
    GGML_ASSERT(backend);
    return backend->device;
}

// backend copy

void ggml_backend_tensor_copy(const struct ggml_tensor * src, struct ggml_tensor * dst) {
    GGML_ASSERT(ggml_are_same_layout(src, dst) && "cannot copy tensors with different layouts");

    if (src == dst) {
        return;
    }

    if (ggml_backend_buffer_is_host(src->buffer)) {
        ggml_backend_tensor_set(dst, src->data, 0, ggml_nbytes(src));
    } else if (ggml_backend_buffer_is_host(dst->buffer)) {
        ggml_backend_tensor_get(src, dst->data, 0, ggml_nbytes(src));
    } else if (!ggml_backend_buffer_copy_tensor(src, dst)) {
#ifndef NDEBUG
        GGML_LOG_DEBUG("%s: warning: slow copy from %s to %s\n", __func__, ggml_backend_buffer_name(src->buffer), ggml_backend_buffer_name(dst->buffer));
#endif // NDEBUG
        size_t nbytes = ggml_nbytes(src);
        void * data = malloc(nbytes);
        ggml_backend_tensor_get(src, data, 0, nbytes);
        ggml_backend_tensor_set(dst, data, 0, nbytes);
        free(data);
    }
}

void ggml_backend_tensor_copy_async(ggml_backend_t backend_src, ggml_backend_t backend_dst, const struct ggml_tensor * src, struct ggml_tensor * dst) {
    GGML_ASSERT(ggml_are_same_layout(src, dst) && "cannot copy tensors with different layouts");

    if (src == dst) {
        return;
    }

    GGML_ASSERT(backend_dst);
    if (backend_dst->iface.cpy_tensor_async != NULL) {
        if (backend_dst->iface.cpy_tensor_async(backend_src, backend_dst, src, dst)) {
            return;
        }
    }

    // an async copy would normally happen after all the queued operations on both backends are completed
    // to simulate the same behavior, we need to synchronize both backends first, and do a blocking copy
    ggml_backend_synchronize(backend_src);
    ggml_backend_synchronize(backend_dst);
    ggml_backend_tensor_copy(src, dst);
}

// events

ggml_backend_event_t ggml_backend_event_new(ggml_backend_dev_t device) {
    // null device is allowed for the transition period to the device interface
    if (device == NULL || device->iface.event_new == NULL) {
        return NULL;
    }
    return device->iface.event_new(device);
}

void ggml_backend_event_free(ggml_backend_event_t event) {
    if (event == NULL) {
        return;
    }
    event->device->iface.event_free(event->device, event);
}

void ggml_backend_event_record(ggml_backend_event_t event, ggml_backend_t backend) {
    GGML_ASSERT(backend);
    GGML_ASSERT(backend->iface.event_record != NULL);

    backend->iface.event_record(backend, event);
}

void ggml_backend_event_synchronize(ggml_backend_event_t event) {
    GGML_ASSERT(event);
    GGML_ASSERT(event->device->iface.event_synchronize);

    event->device->iface.event_synchronize(event->device, event);
}

void ggml_backend_event_wait(ggml_backend_t backend, ggml_backend_event_t event) {
    GGML_ASSERT(backend);
    GGML_ASSERT(backend->iface.event_wait != NULL);

    backend->iface.event_wait(backend, event);
}

static void ggml_backend_graph_optimize(ggml_backend_t backend, struct ggml_cgraph * cgraph) {
    GGML_ASSERT(backend);
    if (backend->iface.graph_optimize != NULL) {
        backend->iface.graph_optimize(backend, cgraph);
    }
}

// Backend device

const char * ggml_backend_dev_name(ggml_backend_dev_t device) {
    GGML_ASSERT(device);
    return device->iface.get_name(device);
}

const char * ggml_backend_dev_description(ggml_backend_dev_t device) {
    GGML_ASSERT(device);
    return device->iface.get_description(device);
}

void ggml_backend_dev_memory(ggml_backend_dev_t device, size_t * free, size_t * total) {
    GGML_ASSERT(device);
    device->iface.get_memory(device, free, total);
}

enum ggml_backend_dev_type ggml_backend_dev_type(ggml_backend_dev_t device) {
    GGML_ASSERT(device);
    return device->iface.get_type(device);
}

void ggml_backend_dev_get_props(ggml_backend_dev_t device, struct ggml_backend_dev_props * props) {
    GGML_ASSERT(device);
    memset(props, 0, sizeof(*props));
    device->iface.get_props(device, props);
}

ggml_backend_reg_t ggml_backend_dev_backend_reg(ggml_backend_dev_t device) {
    GGML_ASSERT(device);
    return device->reg;
}

ggml_backend_t ggml_backend_dev_init(ggml_backend_dev_t device, const char * params) {
    GGML_ASSERT(device);
    return device->iface.init_backend(device, params);
}

ggml_backend_buffer_type_t ggml_backend_dev_buffer_type(ggml_backend_dev_t device) {
    GGML_ASSERT(device);
    return device->iface.get_buffer_type(device);
}

ggml_backend_buffer_type_t ggml_backend_dev_host_buffer_type(ggml_backend_dev_t device) {
    GGML_ASSERT(device);
    if (device->iface.get_host_buffer_type == NULL) {
        return NULL;
    }

    return device->iface.get_host_buffer_type(device);
}

ggml_backend_buffer_t ggml_backend_dev_buffer_from_host_ptr(ggml_backend_dev_t device, void * ptr, size_t size, size_t max_tensor_size) {
    GGML_ASSERT(device);
    return device->iface.buffer_from_host_ptr(device, ptr, size, max_tensor_size);
}

bool ggml_backend_dev_supports_op(ggml_backend_dev_t device, const struct ggml_tensor * op) {
    GGML_ASSERT(device);
    return device->iface.supports_op(device, op);
}

bool ggml_backend_dev_supports_buft(ggml_backend_dev_t device, ggml_backend_buffer_type_t buft) {
    GGML_ASSERT(device);
    return device->iface.supports_buft(device, buft);
}

bool ggml_backend_dev_offload_op(ggml_backend_dev_t device, const struct ggml_tensor * op) {
    GGML_ASSERT(device);
    if (device->iface.offload_op != NULL) {
        return device->iface.offload_op(device, op);
    }

    return false;
}

// Backend (reg)

const char * ggml_backend_reg_name(ggml_backend_reg_t reg) {
    GGML_ASSERT(reg);
    return reg->iface.get_name(reg);
}

size_t ggml_backend_reg_dev_count(ggml_backend_reg_t reg) {
    GGML_ASSERT(reg);
    return reg->iface.get_device_count(reg);
}

ggml_backend_dev_t ggml_backend_reg_dev_get(ggml_backend_reg_t reg, size_t index) {
    GGML_ASSERT(reg);
    return reg->iface.get_device(reg, index);
}

void * ggml_backend_reg_get_proc_address(ggml_backend_reg_t reg, const char * name) {
    GGML_ASSERT(reg);
    if (!reg->iface.get_proc_address) {
        return NULL;
    }
    return reg->iface.get_proc_address(reg, name);
}

// multi-buffer buffer

struct ggml_backend_multi_buffer_context {
    ggml_backend_buffer_t * buffers;
    size_t n_buffers;
};

static void ggml_backend_multi_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(buffer);
    ggml_backend_multi_buffer_context * ctx = (ggml_backend_multi_buffer_context *) buffer->context;
    for (size_t i = 0; i < ctx->n_buffers; i++) {
        ggml_backend_buffer_free(ctx->buffers[i]);
    }

    free(ctx->buffers);
    free(ctx);
}

static void ggml_backend_multi_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    GGML_ASSERT(buffer);
    ggml_backend_multi_buffer_context * ctx = (ggml_backend_multi_buffer_context *) buffer->context;
    for (size_t i = 0; i < ctx->n_buffers; i++) {
        ggml_backend_buffer_clear(ctx->buffers[i], value);
    }
}

static const struct ggml_backend_buffer_i ggml_backend_multi_buffer_i = {
    /* .free_buffer     = */ ggml_backend_multi_buffer_free_buffer,
    /* .get_base        = */ NULL,
    /* .init_tensor     = */ NULL,
    /* .memset_tensor   = */ NULL,
    /* .set_tensor      = */ NULL,
    /* .get_tensor      = */ NULL,
    /* .set_tensor_2d   = */ NULL,
    /* .get_tensor_2d   = */ NULL,
    /* .cpy_tensor      = */ NULL,
    /* .clear           = */ ggml_backend_multi_buffer_clear,
    /* .reset           = */ NULL,
};

ggml_backend_buffer_t ggml_backend_multi_buffer_alloc_buffer(ggml_backend_buffer_t * buffers, size_t n_buffers) {
    ggml_backend_multi_buffer_context * ctx = (ggml_backend_multi_buffer_context *) malloc(sizeof(struct ggml_backend_multi_buffer_context));
    ctx->n_buffers = n_buffers;
    ctx->buffers = (ggml_backend_buffer_t *) malloc(n_buffers * sizeof(ggml_backend_buffer_t));

    GGML_ASSERT(ctx->buffers != NULL);

    size_t total_size = 0;
    for (size_t i = 0; i < n_buffers; i++) {
        ctx->buffers[i] = buffers[i];
        total_size += ggml_backend_buffer_get_size(buffers[i]);
    }

    return ggml_backend_buffer_init(buffers[0]->buft, ggml_backend_multi_buffer_i, ctx, total_size);
}

bool ggml_backend_buffer_is_multi_buffer(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(buffer);
    return buffer->iface.free_buffer == ggml_backend_multi_buffer_free_buffer;
}

void ggml_backend_multi_buffer_set_usage(ggml_backend_buffer_t buffer, enum ggml_backend_buffer_usage usage) {
    GGML_ASSERT(buffer);
    GGML_ASSERT(ggml_backend_buffer_is_multi_buffer(buffer));
    ggml_backend_multi_buffer_context * ctx = (ggml_backend_multi_buffer_context *) buffer->context;
    for (size_t i = 0; i < ctx->n_buffers; i++) {
        ggml_backend_buffer_set_usage(ctx->buffers[i], usage);
    }
}

// creates a copy of the tensor with the same memory layout
static struct ggml_tensor * ggml_dup_tensor_layout(struct ggml_context * ctx, const struct ggml_tensor * tensor) {
    struct ggml_tensor * dup = ggml_dup_tensor(ctx, tensor);
    for (int i = 0; i < GGML_MAX_DIMS; i++) {
        dup->nb[i] = tensor->nb[i];
    }
    return dup;
}

static bool ggml_is_view_op(enum ggml_op op) {
    return op == GGML_OP_VIEW || op == GGML_OP_RESHAPE || op == GGML_OP_PERMUTE || op == GGML_OP_TRANSPOSE;
}

// scheduler

#ifndef GGML_SCHED_MAX_BACKENDS
#define GGML_SCHED_MAX_BACKENDS 16
#endif

#ifndef GGML_SCHED_MAX_SPLIT_INPUTS
#define GGML_SCHED_MAX_SPLIT_INPUTS 30
#endif

#ifndef GGML_SCHED_MAX_COPIES
#define GGML_SCHED_MAX_COPIES 4
#endif

#ifndef GGML_SCHED_MAX_STAGING_SLOTS
#define GGML_SCHED_MAX_STAGING_SLOTS 4
#endif

struct ggml_backend_sched_split {
    int backend_id;
    int copy_id;
    int staging_id;
    size_t staging_used;
    int i_start;
    int i_end;
    struct ggml_tensor ** inputs;
    int n_inputs;
    int inputs_capacity;
    // graph view of this split
    struct ggml_cgraph graph;
};

struct ggml_backend_sched_weight_cache_entry {
    const struct ggml_tensor * src;
    size_t offset;
    bool valid;
};

struct ggml_backend_sched_gate_ram_mirror_entry {
    struct ggml_tensor * src;
    void * data;
    size_t size;
};

struct ggml_backend_sched_gate_wc_mirror_entry {
    const struct ggml_tensor * src;
    int64_t n_rows;
    ggml_backend_buffer_t buffer;
    struct ggml_tensor alias;
};

// Persistent metadata for the cooperative CPU/GPU FFN-gate path.  The tensor
// payloads live in scheduler-owned reusable buffers; keeping the ggml context
// and graphs alive avoids rebuilding the same two one-node graphs for every
// layer and every generated token.
struct ggml_backend_sched_gate_row_split_plan {
    struct ggml_context * ctx;
    struct ggml_tensor * weight_cpu;
    struct ggml_tensor * input_cpu;
    struct ggml_tensor * output_cpu;
    struct ggml_tensor * weight_gpu[2];
    struct ggml_tensor * input_gpu;
    struct ggml_tensor * output_gpu[2];
    struct ggml_cgraph * graph_cpu;
    struct ggml_cgraph * graph_gpu[2];
    int64_t n_in;
    int64_t n_out;
    int64_t n_batch;
    int64_t n_gpu;
};

struct ggml_backend_sched_gate_row_split_pending {
    bool active;
    struct ggml_backend_sched_gate_row_split_plan * plan;
    struct ggml_tensor * output;
    ggml_backend_t gpu_backend;
    enum ggml_status gpu_status;
    int gpu_percent;
    int64_t n_gpu;
    int64_t n_cpu;
    int64_t n_batch;
    size_t gpu_output_bytes;
    int64_t begin_us;
    int slot;
};

struct ggml_backend_sched_gate_row_split_prefetch {
    const struct ggml_tensor * weight;
    int64_t n_gpu;
    int slot;
};

// A CPU backend compute call is synchronous.  Keep one persistent worker per
// scheduler so a host FFN gate can run while the CUDA backend computes the
// independent FFN-up branch.  The scheduler still joins the worker before the
// first node which consumes the gate, so this only changes scheduling, not the
// graph or its numerical operations.
struct ggml_backend_sched_cpu_gpu_overlap {
    std::mutex mutex;
    std::condition_variable task_cv;
    std::condition_variable done_cv;
    bool stop = false;
    bool task_ready = false;
    bool task_done = true;
    ggml_backend_t backend = nullptr;
    struct ggml_cgraph graph = {};
    enum ggml_status status = GGML_STATUS_SUCCESS;
    std::thread worker;

    ggml_backend_sched_cpu_gpu_overlap() : worker([this]() {
        for (;;) {
            ggml_backend_t task_backend = nullptr;
            struct ggml_cgraph task_graph = {};
            {
                std::unique_lock<std::mutex> lock(mutex);
                task_cv.wait(lock, [this]() { return stop || task_ready; });
                if (stop && !task_ready) {
                    return;
                }
                task_backend = backend;
                task_graph = graph;
                task_ready = false;
            }

            const enum ggml_status task_status =
                    ggml_backend_graph_compute_async(task_backend, &task_graph);

            {
                std::lock_guard<std::mutex> lock(mutex);
                status = task_status;
                task_done = true;
            }
            done_cv.notify_one();
        }
    }) {}

    ~ggml_backend_sched_cpu_gpu_overlap() {
        {
            std::lock_guard<std::mutex> lock(mutex);
            stop = true;
        }
        task_cv.notify_one();
        if (worker.joinable()) {
            worker.join();
        }
    }

    void submit(ggml_backend_t task_backend, const struct ggml_cgraph & task_graph) {
        {
            std::lock_guard<std::mutex> lock(mutex);
            GGML_ASSERT(task_done && !task_ready);
            backend = task_backend;
            graph = task_graph;
            status = GGML_STATUS_SUCCESS;
            task_done = false;
            task_ready = true;
        }
        task_cv.notify_one();
    }

    enum ggml_status wait() {
        std::unique_lock<std::mutex> lock(mutex);
        done_cv.wait(lock, [this]() { return task_done; });
        return status;
    }
};

struct ggml_backend_sched {
    bool is_reset; // true if the scheduler has been reset since the last graph split
    bool is_alloc;

    int n_backends;

    ggml_backend_t backends[GGML_SCHED_MAX_BACKENDS];
    ggml_backend_buffer_type_t bufts[GGML_SCHED_MAX_BACKENDS];
    ggml_gallocr_t galloc;

    // hash map of the nodes in the graph
    struct ggml_hash_set  hash_set;
    int                 * hv_tensor_backend_ids; // [hash_set.size]
    struct ggml_tensor ** hv_tensor_copies;      // [hash_set.size][n_backends][n_copies]

    int * node_backend_ids; // [graph_size]
    int * leaf_backend_ids; // [graph_size]

    int * prev_node_backend_ids; // [graph_size]
    int * prev_leaf_backend_ids; // [graph_size]

    // copy of the graph with modified inputs
    struct ggml_cgraph graph;

    // graph splits
    struct ggml_backend_sched_split * splits;
    int n_splits;
    int splits_capacity;

    // pipeline parallelism support
    int n_copies;
    bool staging_double_buffer;
    int staging_slots;
    int cur_copy;
    int next_copy;
    ggml_backend_event_t events[GGML_SCHED_MAX_BACKENDS][GGML_SCHED_MAX_COPIES];
    ggml_backend_event_t staging_events[GGML_SCHED_MAX_BACKENDS][GGML_SCHED_MAX_STAGING_SLOTS];
    bool staging_event_recorded[GGML_SCHED_MAX_BACKENDS][GGML_SCHED_MAX_STAGING_SLOTS];
    struct ggml_tensor * staging_roots[GGML_SCHED_MAX_BACKENDS][GGML_SCHED_MAX_STAGING_SLOTS];
    ggml_backend_buffer_t staging_buffers[GGML_SCHED_MAX_BACKENDS][GGML_SCHED_MAX_STAGING_SLOTS];
    struct ggml_tensor * weight_cache_roots[GGML_SCHED_MAX_BACKENDS];
    ggml_backend_buffer_t weight_cache_buffers[GGML_SCHED_MAX_BACKENDS];
    struct ggml_backend_sched_weight_cache_entry * weight_cache_entries;
    int weight_cache_n;
    int weight_cache_capacity;
    size_t weight_cache_used;
    bool weight_cache_prefill;
    bool weight_cache_phase_set;
    bool weight_cache_owner;
    struct ggml_backend_sched_gate_ram_mirror_entry * gate_ram_mirrors;
    int gate_ram_mirror_n;
    int gate_ram_mirror_capacity;
    struct ggml_backend_sched_gate_wc_mirror_entry * gate_wc_mirrors;
    int gate_wc_mirror_n;
    int gate_wc_mirror_capacity;
    ggml_backend_buffer_t gate_split_gpu_weight_buffer[2];
    ggml_backend_buffer_t gate_split_gpu_output_buffer;
    ggml_backend_buffer_t gate_split_cpu_output_buffer;
    size_t gate_split_gpu_weight_capacity[2];
    size_t gate_split_gpu_output_capacity;
    size_t gate_split_cpu_output_capacity;
    void * gate_split_gpu_output_host;
    size_t gate_split_gpu_output_host_capacity;
    struct ggml_backend_sched_gate_row_split_plan * gate_split_plans[32];
    struct ggml_backend_sched_cpu_gpu_overlap * cpu_gpu_overlap;
    // Optional phase-local cache which aliases output.weight's existing GPU
    // allocation. During prompt chunks which do not compute logits, the output
    // matrix is dead storage and can hold host-resident FFN gates. The original
    // bytes are restored before a graph which consumes output.weight and remain
    // resident for decode. This avoids allocating a second ~1 GiB cache.
    struct ggml_tensor * weight_cache_alias_output;
    void * weight_cache_alias_shadow;
    size_t weight_cache_alias_size;
    bool weight_cache_alias_dirty;
    bool weight_cache_alias_can_fill;
    // Speculative verification may build multi-token graphs during decode.
    // Once the first real prompt has populated and restored the alias, do not
    // mistake those graphs for another prompt and overwrite output.weight.
    bool weight_cache_alias_completed;
    struct ggml_tensor ** graph_inputs;
    int n_graph_inputs;
    int graph_inputs_capacity;

    struct ggml_context * ctx;

    ggml_backend_sched_eval_callback callback_eval;
    void * callback_eval_user_data;

    char * context_buffer;
    size_t context_buffer_size;

    bool op_offload;

    int debug;

    // used for debugging graph reallocations [GGML_SCHED_DEBUG_REALLOC]
    // ref: https://github.com/ggml-org/llama.cpp/pull/17617
    int debug_realloc;
    int debug_graph_size;
    int debug_prev_graph_size;
};

#define hash_id(tensor) ggml_hash_find_or_insert(&sched->hash_set, tensor)
#define tensor_backend_id(tensor) sched->hv_tensor_backend_ids[hash_id(tensor)]
#define tensor_id_copy(id, backend_id, copy_id) sched->hv_tensor_copies[(id) * sched->n_backends * sched->n_copies + (backend_id) * sched->n_copies + (copy_id)]
#define tensor_copy(tensor, backend_id, copy_id) tensor_id_copy(hash_id(tensor), backend_id, copy_id)

static bool ggml_backend_sched_staging_double_buffer_enabled(void) {
    const char * value = getenv("LLAMA_STAGING_DOUBLE_BUFFER");
    return value != NULL && value[0] != '\0' && atoi(value) != 0;
}

static size_t ggml_backend_sched_staging_buffer_size(void) {
    size_t size = 64ull*1024*1024;
    if (const char * value = getenv("LLAMA_STAGING_BUFFER_MIB")) {
        char * end = NULL;
        const unsigned long requested = strtoul(value, &end, 10);
        if (end != value && *end == '\0' && requested > 0) {
            size = (size_t) requested*1024*1024;
        }
    }
    return size;
}

static bool ggml_backend_sched_staging_device_reuse_wait_enabled(void) {
    const char * value = getenv("LLAMA_STAGING_DEVICE_REUSE_WAIT");
    return value != NULL && value[0] != '\0' && atoi(value) != 0;
}

static size_t ggml_backend_sched_weight_cache_size(void) {
    const char * value = getenv("LLAMA_WEIGHT_CACHE_MIB");
    if (value == NULL || value[0] == '\0') {
        return 0;
    }
    char * end = NULL;
    const unsigned long requested = strtoul(value, &end, 10);
    if (end == value || *end != '\0' || requested == 0) {
        return 0;
    }
    return (size_t) requested*1024*1024;
}

static bool ggml_backend_sched_weight_cache_alias_output_enabled(void) {
    const char * value = getenv("LLAMA_WEIGHT_CACHE_ALIAS_OUTPUT");
    return value != NULL && value[0] != '\0' && strcmp(value, "0") != 0;
}

static bool ggml_backend_sched_weight_cache_decode_gates_enabled(void) {
    const char * value = getenv("LLAMA_WEIGHT_CACHE_DECODE_GATES");
    return value != NULL && value[0] != '\0' && strcmp(value, "0") != 0;
}

static bool ggml_backend_sched_weight_cache_decode_gate_fits(
        const struct ggml_tensor * tensor) {
    if (tensor == NULL) {
        return false;
    }
    int block = -1;
    if (sscanf(tensor->name, "blk.%d.ffn_gate.weight", &block) != 1 || block < 0) {
        return false;
    }
    const size_t alignment = 256;
    const size_t bytes = (ggml_nbytes(tensor) + alignment - 1) & ~(alignment - 1);
    return bytes > 0 && ((size_t) block + 1)*bytes <= ggml_backend_sched_weight_cache_size();
}

static int ggml_backend_sched_weight_cache_alias_min_block(void) {
    const char * value = getenv("LLAMA_WEIGHT_CACHE_ALIAS_MIN_BLOCK");
    if (value == NULL || value[0] == '\0') {
        return 0;
    }
    return std::max(0, atoi(value));
}

static struct ggml_tensor * ggml_backend_sched_weight_cache_root(
        ggml_backend_sched_t sched, int backend_id) {
    struct ggml_tensor * & root = sched->weight_cache_roots[backend_id];
    if (root == NULL) {
        root = ggml_new_tensor_1d(sched->ctx, GGML_TYPE_I8,
                (int64_t) ggml_backend_sched_weight_cache_size());
        ggml_format_name(root, "weight_cache_root_%d", backend_id);
        ggml_set_input(root);
        ggml_set_output(root);
        tensor_backend_id(root) = backend_id;

        ggml_backend_buffer_t & buffer = sched->weight_cache_buffers[backend_id];
        if (buffer == NULL) {
            const size_t alloc_size = ggml_backend_buft_get_alloc_size(sched->bufts[backend_id], root);
            buffer = ggml_backend_buft_alloc_buffer(sched->bufts[backend_id], alloc_size);
            GGML_ASSERT(buffer != NULL);
            ggml_backend_buffer_set_usage(buffer, GGML_BACKEND_BUFFER_USAGE_COMPUTE);
        }
        GGML_ASSERT(ggml_backend_tensor_alloc(buffer, root,
                ggml_backend_buffer_get_base(buffer)) == GGML_STATUS_SUCCESS);
    }
    return root;
}

static bool ggml_backend_sched_weight_cache_name_eligible(
        ggml_backend_sched_t sched, const struct ggml_tensor * tensor) {
    if (!sched->weight_cache_owner || ggml_backend_sched_weight_cache_size() == 0 || tensor == NULL) {
        return false;
    }
    if (ggml_backend_sched_weight_cache_alias_output_enabled()) {
        int block = -1;
        const bool is_gate = sscanf(tensor->name, "blk.%d.ffn_gate.weight", &block) == 1;
        return sched->weight_cache_alias_can_fill &&
               is_gate &&
               block >= ggml_backend_sched_weight_cache_alias_min_block();
    }
    const bool is_gate = strstr(tensor->name, ".ffn_gate.weight") != NULL &&
                         strncmp(tensor->name, "blk.", 4) == 0;
    if (ggml_backend_sched_weight_cache_decode_gates_enabled()) {
        // Keep prompt evaluation on the proven staging pipeline.  During
        // decode, persist as many early host gates as fit and reuse them
        // across token graphs instead of copying them every cycle.
        const bool eligible = !sched->weight_cache_prefill &&
                              (is_gate || strcmp(tensor->name, "output.weight") == 0);
        if (is_gate && getenv("LLAMA_WEIGHT_CACHE_TRACE") != NULL) {
            fprintf(stderr, "WEIGHT_CACHE_CHECK: %s owner=%d prefill=%d eligible=%d\n",
                    tensor->name, (int) sched->weight_cache_owner,
                    (int) sched->weight_cache_prefill, (int) eligible);
            fflush(stderr);
        }
        return eligible;
    }
    if (sched->weight_cache_prefill) {
        return is_gate ||
               strcmp(tensor->name, "output.weight") == 0;
    }
    return strcmp(tensor->name, "output.weight") == 0;
}

static struct ggml_backend_sched_weight_cache_entry * ggml_backend_sched_weight_cache_find(
        ggml_backend_sched_t sched, const struct ggml_tensor * src) {
    for (int i = 0; i < sched->weight_cache_n; ++i) {
        if (sched->weight_cache_entries[i].src == src) {
            return &sched->weight_cache_entries[i];
        }
    }
    return NULL;
}

static struct ggml_backend_sched_weight_cache_entry * ggml_backend_sched_weight_cache_get_or_add(
        ggml_backend_sched_t sched, const struct ggml_tensor * src) {
    struct ggml_backend_sched_weight_cache_entry * entry =
            ggml_backend_sched_weight_cache_find(sched, src);
    if (entry != NULL) {
        return entry;
    }
    const size_t alignment = 256;
    const size_t offset = (sched->weight_cache_used + alignment - 1) & ~(alignment - 1);
    size_t capacity = ggml_backend_sched_weight_cache_size();
    if (ggml_backend_sched_weight_cache_alias_output_enabled() && sched->weight_cache_alias_output != NULL) {
        capacity = std::min(capacity, ggml_nbytes(sched->weight_cache_alias_output));
    }
    // During prefill, output.weight is consumed only after all backbone gates.
    // Let it overlap the entire dedicated cache at offset zero.  The execution
    // path invalidates gate entries before filling output, and invalidates the
    // output entry when a later prompt graph refills a gate.  This turns one
    // physical ~1 GiB allocation into a phase-local FFN/output cache without
    // changing either tensor's bytes.
    if (sched->weight_cache_prefill && strcmp(src->name, "output.weight") == 0) {
        if (sched->weight_cache_n >= sched->weight_cache_capacity || ggml_nbytes(src) > capacity) {
            return NULL;
        }
        entry = &sched->weight_cache_entries[sched->weight_cache_n++];
        entry->src = src;
        entry->offset = 0;
        entry->valid = false;
        return entry;
    }
    if (sched->weight_cache_n >= sched->weight_cache_capacity ||
        offset + ggml_nbytes(src) > capacity) {
        return NULL;
    }
    entry = &sched->weight_cache_entries[sched->weight_cache_n++];
    entry->src = src;
    entry->offset = offset;
    entry->valid = false;
    sched->weight_cache_used = offset + ggml_nbytes(src);
    return entry;
}

static struct ggml_tensor * ggml_backend_sched_staging_root(
        ggml_backend_sched_t sched, int backend_id, int copy_id) {
    GGML_ASSERT(copy_id >= 0 && copy_id < sched->staging_slots);
    struct ggml_tensor * & root = sched->staging_roots[backend_id][copy_id];
    if (root == NULL) {
        root = ggml_new_tensor_1d(sched->ctx, GGML_TYPE_I8,
                (int64_t) ggml_backend_sched_staging_buffer_size());
        ggml_format_name(root, "staging_root_%d_%d", backend_id, copy_id);
        ggml_set_input(root);
        ggml_set_output(root); // keep both ping-pong buffers alive for the full graph
        tensor_backend_id(root) = backend_id;

        ggml_backend_buffer_t & buffer = sched->staging_buffers[backend_id][copy_id];
        if (buffer == NULL) {
            const size_t alloc_size = ggml_backend_buft_get_alloc_size(sched->bufts[backend_id], root);
            buffer = ggml_backend_buft_alloc_buffer(sched->bufts[backend_id], alloc_size);
            GGML_ASSERT(buffer != NULL);
            ggml_backend_buffer_set_usage(buffer, GGML_BACKEND_BUFFER_USAGE_COMPUTE);
        }
        GGML_ASSERT(ggml_backend_tensor_alloc(buffer, root, ggml_backend_buffer_get_base(buffer)) == GGML_STATUS_SUCCESS);
    }
    return root;
}

static void ggml_backend_sched_split_inputs_grow(struct ggml_backend_sched_split * split) {
    int new_cap = GGML_SCHED_MAX_SPLIT_INPUTS;
    if (split->inputs_capacity > 0) {
        new_cap = 2*split->inputs_capacity;
        GGML_LOG_WARN("%s: increasing split inputs capacity from %d to %d\n", __func__, split->inputs_capacity, new_cap);
    }
    auto * pnew = (struct ggml_tensor **) realloc((void *) split->inputs, new_cap * sizeof(struct ggml_tensor *));
    if (pnew == NULL) {
        GGML_LOG_ERROR("%s: failed to allocate %zu bytes\n", __func__, new_cap * sizeof(struct ggml_tensor *));
        GGML_ABORT("failed to grow split inputs container");
    }
    split->inputs = pnew;
    split->inputs_capacity = new_cap;
}

static void ggml_backend_sched_graph_inputs_grow(ggml_backend_sched_t sched) {
    int new_cap = GGML_SCHED_MAX_SPLIT_INPUTS;
    if (sched->graph_inputs_capacity > 0) {
        new_cap = 2*sched->graph_inputs_capacity;
        GGML_LOG_WARN("%s: increasing graph inputs capacity from %d to %d\n", __func__, sched->graph_inputs_capacity, new_cap);
    }
    auto * pnew = (struct ggml_tensor **) realloc((void *) sched->graph_inputs, new_cap * sizeof(struct ggml_tensor *));
    if (pnew == NULL) {
        GGML_LOG_ERROR("%s: failed to allocate %zu bytes\n", __func__, new_cap * sizeof(struct ggml_tensor *));
        GGML_ABORT("failed to grow graph inputs container");
    }
    sched->graph_inputs = pnew;
    sched->graph_inputs_capacity = new_cap;
}

// returns the priority of the backend, lower id is higher priority
static int ggml_backend_sched_backend_id(ggml_backend_sched_t sched, ggml_backend_t backend) {
    for (int i = 0; i < sched->n_backends; i++) {
        if (sched->backends[i] == backend) {
            return i;
        }
    }
    return -1;
}

// A CUDA_Host weight can either be consumed directly by CUDA over PCIe or be
// copied into a temporary CUDA buffer.  Direct access is useful for the tiny
// decode batches because it avoids making a new copy for every token, while a
// contiguous copy is substantially faster for large prompt batches.  Keep the
// policy in the scheduler so the same mapped allocation can use both paths.
//
// LLAMA_CUDA_HOST_DIRECT_MAX_BATCH=N means:
//   batch <= N: CUDA consumes CUDA_Host weights directly;
//   batch >  N: treat the weight as CPU-resident, allowing op_offload to make
//               the normal temporary CUDA copy.
static int64_t ggml_backend_sched_cuda_host_direct_max_batch(void) {
    const char * value = getenv("LLAMA_CUDA_HOST_DIRECT_MAX_BATCH");
    if (value == NULL || value[0] == '\0') {
        return -1;
    }
    return strtoll(value, NULL, 10);
}

// Optional inverse threshold used by memory-constrained prefill profiles.
// A large prompt batch can consume a mapped CUDA_Host weight directly while
// tiny decode batches keep using a temporary device copy.  This avoids
// reserving the large prefill staging graph without forcing latency-sensitive
// decode to read weights over PCIe.
static int64_t ggml_backend_sched_cuda_host_direct_min_batch(void) {
    const char * value = getenv("LLAMA_CUDA_HOST_DIRECT_MIN_BATCH");
    if (value == NULL || value[0] == '\0') {
        return -1;
    }
    return strtoll(value, NULL, 10);
}

// Small mapped weights are cheap enough to read directly over PCIe and, more
// importantly for close-fit profiles, do not justify a persistent device-side
// copy.  Large FFN weights still follow the batch policy above.
static size_t ggml_backend_sched_cuda_host_direct_max_bytes(void) {
    const char * value = getenv("LLAMA_CUDA_HOST_DIRECT_MAX_MIB");
    if (value == NULL || value[0] == '\0') {
        return 0;
    }
    char * end = NULL;
    const unsigned long requested = strtoul(value, &end, 10);
    if (end == value || *end != '\0' || requested == 0) {
        return 0;
    }
    return (size_t) requested*1024*1024;
}

static int64_t ggml_backend_sched_op_batch_size(const struct ggml_tensor * op) {
    if (op == NULL) {
        return 0;
    }
    switch (op->op) {
        case GGML_OP_GET_ROWS:
            return 0;
        case GGML_OP_MUL_MAT:
            return op->ne[1];
        case GGML_OP_MUL_MAT_ID:
        case GGML_OP_ROPE:
        case GGML_OP_ROPE_BACK:
            return op->ne[2];
        default:
            return ggml_nrows(op);
    }
}

static bool ggml_backend_sched_is_cuda_host_buffer(ggml_backend_buffer_t buffer) {
    return buffer != NULL && strstr(ggml_backend_buffer_name(buffer), "CUDA_Host") != NULL;
}

static const struct ggml_tensor * ggml_backend_sched_view_root(const struct ggml_tensor * tensor) {
    while (tensor != NULL && tensor->view_src != NULL) {
        tensor = tensor->view_src;
    }
    return tensor;
}

static bool ggml_backend_sched_is_staged_kv_input(
        const struct ggml_tensor * tensor,
        const struct ggml_tensor * consumer) {
    if (getenv("LLAMA_KV_HOST_STAGE") == NULL ||
        tensor == NULL || consumer == NULL || consumer->op != GGML_OP_FLASH_ATTN_EXT) {
        return false;
    }

    const struct ggml_tensor * root = ggml_backend_sched_view_root(tensor);
    return root != NULL &&
           (strncmp(root->name, "cache_k_l", 9) == 0 || strncmp(root->name, "cache_v_l", 9) == 0);
}

static bool ggml_backend_sched_cuda_host_use_direct(
        const struct ggml_tensor * tensor,
        const struct ggml_tensor * op) {
    const size_t max_bytes = ggml_backend_sched_cuda_host_direct_max_bytes();
    if (tensor != NULL && max_bytes > 0 && ggml_nbytes(tensor) <= max_bytes) {
        return true;
    }
    const int64_t batch = ggml_backend_sched_op_batch_size(op);
    const int64_t max_batch = ggml_backend_sched_cuda_host_direct_max_batch();
    const int64_t min_batch = ggml_backend_sched_cuda_host_direct_min_batch();
    return (max_batch >= 0 && batch <= max_batch) ||
           (min_batch >= 0 && batch >= min_batch);
}

static bool ggml_backend_sched_cuda_host_force_copy(
        ggml_backend_sched_t sched,
        ggml_backend_buffer_t buffer,
        const struct ggml_tensor * tensor,
        const struct ggml_tensor * consumer,
        int backend_id) {
    const bool dynamic_cuda_host =
           ggml_backend_sched_cuda_host_direct_max_batch() >= 0 ||
           ggml_backend_sched_cuda_host_direct_min_batch() >= 0 ||
           ggml_backend_sched_cuda_host_direct_max_bytes() > 0;
    return dynamic_cuda_host &&
           backend_id < sched->n_backends - 1 &&
           ggml_backend_sched_is_cuda_host_buffer(buffer) &&
           consumer != NULL && consumer->op != GGML_OP_NONE &&
           !ggml_backend_sched_cuda_host_use_direct(tensor, consumer);
}

static int ggml_backend_sched_backend_from_buffer(ggml_backend_sched_t sched, const struct ggml_tensor * tensor, const struct ggml_tensor * op) {
    ggml_backend_buffer_t buffer = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;
    if (buffer == NULL) {
        return -1;
    }

    const bool dynamic_cuda_host =
        (ggml_backend_sched_cuda_host_direct_max_batch() >= 0 ||
         ggml_backend_sched_cuda_host_direct_min_batch() >= 0 ||
         ggml_backend_sched_cuda_host_direct_max_bytes() > 0) &&
        ggml_backend_sched_is_cuda_host_buffer(buffer);

    // find highest prio backend that supports the buffer type and the op
    for (int i = 0; i < sched->n_backends; i++) {
        // Keep CUDA_Host leaves associated with the CPU backend.  A decode op
        // can still consume the original mapped pointer directly, while a
        // prefill op can request a temporary CUDA copy below.
        if (dynamic_cuda_host && i < sched->n_backends - 1 &&
            (op->op == GGML_OP_NONE || !ggml_backend_sched_cuda_host_use_direct(tensor, op))) {
            continue;
        }
        if (ggml_backend_supports_buft(sched->backends[i], buffer->buft) &&
            ggml_backend_supports_op(sched->backends[i], op)) {
            return i;
        }
    }

#ifndef NDEBUG
    GGML_LOG_DEBUG("%s: warning: no backend supports op %s with a weight with buffer type %s used in tensor %s, the weight will need to be copied\n",
        __func__, ggml_op_desc(tensor), ggml_backend_buffer_name(buffer), tensor->name);
#endif

    return -1;
}

#if 0
#define GGML_SCHED_MAX_SPLITS_DEBUG 4096
static char causes[GGML_DEFAULT_GRAPH_SIZE*16 + GGML_SCHED_MAX_SPLITS_DEBUG*GGML_SCHED_MAX_SPLIT_INPUTS][128]; // debug only
#define SET_CAUSE(node, ...) sprintf(causes[hash_id(node)], __VA_ARGS__)
#define GET_CAUSE(node) causes[hash_id(node)]
#else
#define SET_CAUSE(node, ...)
#define GET_CAUSE(node) ""
#endif

// returns the backend that should be used for the node based on the current locations
static int ggml_backend_sched_backend_id_from_cur(ggml_backend_sched_t sched, struct ggml_tensor * tensor) {
    // assign pre-allocated nodes to their backend
    int cur_backend_id = ggml_backend_sched_backend_from_buffer(sched, tensor, tensor);
    if (cur_backend_id != -1) {
        SET_CAUSE(tensor, "1.dst");
        return cur_backend_id;
    }

    // view_src
    if (tensor->view_src != NULL) {
        cur_backend_id = ggml_backend_sched_backend_from_buffer(sched, tensor->view_src, tensor);
        if (cur_backend_id != -1) {
            SET_CAUSE(tensor, "1.vsrc");
            return cur_backend_id;
        }
    }

    if (tensor->buffer || (tensor->view_src && tensor->view_src->buffer)) {
        // since the tensor is pre-allocated, it cannot be moved to another backend
        ggml_backend_buffer_t buffer = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;
        GGML_ABORT("pre-allocated tensor (%s) in a buffer (%s) that cannot run the operation (%s)", tensor->name, ggml_backend_buffer_name(buffer), ggml_op_name(tensor->op));
    }

    // graph input
    if (tensor->flags & GGML_TENSOR_FLAG_INPUT) {
        cur_backend_id = sched->n_backends - 1; // last backend (assumed CPU)
        SET_CAUSE(tensor, "1.inp");
        return cur_backend_id;
    }

    // operations with weights are preferably run on the same backend as the weights
    // TODO: there are exceptions (see below) - not an ideal solution
    bool allow = true;

    // skip ROPE since the rope freqs tensor is too small to choose a backend based on it
    allow = allow && tensor->op != GGML_OP_ROPE;

    // skip FLASH_ATTN_EXT since the sinks tensor is too small to choose a based based on it
    allow = allow && tensor->op != GGML_OP_FLASH_ATTN_EXT;

    if (allow) {
        for (int i = 0; i < GGML_MAX_SRC; i++) {
            const struct ggml_tensor * src = tensor->src[i];
            if (src == NULL) {
                continue;
            }
            if (src->buffer != NULL && src->buffer->usage == GGML_BACKEND_BUFFER_USAGE_WEIGHTS) {
                int src_backend_id = ggml_backend_sched_backend_from_buffer(sched, src, tensor);
                // check if a backend with higher prio wants to offload the op
                if (sched->op_offload && src_backend_id == sched->n_backends - 1 && ggml_backend_buffer_is_host(src->buffer)) {
                    for (int b = 0; b < src_backend_id; b++) {
                        if (ggml_backend_supports_op(sched->backends[b], tensor) && ggml_backend_offload_op(sched->backends[b], tensor)) {
                            SET_CAUSE(tensor, "1.off");
                            return b;
                        }
                    }
                }
                SET_CAUSE(tensor, "1.wgt%d", i);
                return src_backend_id;
            }
        }
    }

    return -1;
}

static char * fmt_size(size_t size) {
    static char buffer[128];
    if (size >= 1024*1024) {
        snprintf(buffer, sizeof(buffer), "%zuM", size/1024/1024);
    } else {
        snprintf(buffer, sizeof(buffer), "%zuK", size/1024);
    }
    return buffer;
}

static void ggml_backend_sched_print_assignments(ggml_backend_sched_t sched, struct ggml_cgraph * graph) {
    int cur_split = 0;
    for (int i = 0; i < graph->n_nodes; i++) {
        if (cur_split < sched->n_splits && i == sched->splits[cur_split].i_start) {
            ggml_backend_t split_backend = sched->backends[sched->splits[cur_split].backend_id];
            GGML_LOG_DEBUG("\n## SPLIT #%d: %s # %d inputs", cur_split, ggml_backend_name(split_backend),
                sched->splits[cur_split].n_inputs);
            for (int j = 0; j < sched->splits[cur_split].n_inputs; j++) {
                if (j == 0) {
                    GGML_LOG_DEBUG(": ");
                }
                GGML_LOG_DEBUG("[%s (%5.5s)] ", sched->splits[cur_split].inputs[j]->name,
                    fmt_size(ggml_nbytes(sched->splits[cur_split].inputs[j])));
            }
            GGML_LOG_DEBUG("\n");
            cur_split++;
        }
        struct ggml_tensor * node = graph->nodes[i];
        if (ggml_is_view_op(node->op)) {
            continue;
        }
        if (sched->debug > 1) {
            ggml_backend_t tensor_backend = ggml_backend_sched_get_tensor_backend(sched, node);
            GGML_LOG_DEBUG("node #%3d (%10.10s): %20.20s (%5.5s) [%5.5s %8.8s] use=%d,c=%d:", i, ggml_op_desc(node), node->name,
                fmt_size(ggml_nbytes(node)), tensor_backend ? ggml_backend_name(tensor_backend) : "NULL", GET_CAUSE(node),
                graph->use_counts[ggml_hash_find(&graph->visited_hash_set, node)], node->flags & GGML_TENSOR_FLAG_COMPUTE ? 1 : 0);
            for (int j = 0; j < GGML_MAX_SRC; j++) {
                struct ggml_tensor * src = node->src[j];
                if (src == NULL) {
                    continue;
                }
                ggml_backend_t src_backend = ggml_backend_sched_get_tensor_backend(sched, src);
                GGML_LOG_DEBUG(" %20.20s (%5.5s) [%5.5s %8.8s]", src->name,
                    fmt_size(ggml_nbytes(src)), src_backend ? ggml_backend_name(src_backend) : "NULL", GET_CAUSE(src));
            }
            GGML_LOG_DEBUG("\n");
        }
    }
}

static bool ggml_backend_sched_buffer_supported(
        ggml_backend_sched_t sched,
        struct ggml_tensor * t,
        int backend_id,
        const struct ggml_tensor * consumer = NULL) {
    ggml_backend_buffer_t buf = t->view_src ? t->view_src->buffer : t->buffer;
    ggml_backend_buffer_type_t buft = NULL;

    if (buf) {
        // the tensor is already allocated
        buft = buf->buft;
    } else {
        // see if the tensor already has a backend assigned, and use the buffer type of that backend
        int tensor_backend_id = tensor_backend_id(t);
        if (tensor_backend_id == -1 && t->view_src) {
            tensor_backend_id = tensor_backend_id(t->view_src);
        }
        if (tensor_backend_id != -1) {
            buft = sched->bufts[tensor_backend_id];
        }
    }

    if (backend_id < sched->n_backends - 1 &&
        buf != NULL && ggml_backend_buffer_is_host(buf) &&
        ggml_backend_sched_weight_cache_name_eligible(sched, t)) {
        return false;
    }

    if (ggml_backend_sched_cuda_host_force_copy(sched, buf, t, consumer, backend_id)) {
        return false;
    }

    // A host-resident KV cache can retain the full configured context in RAM
    // while only the active K/V view is copied to a small reusable CUDA
    // staging buffer for Flash Attention.  KV stores continue to use the
    // mapped host allocation directly; only FA reads are forced through the
    // scheduler copy path.
    if (backend_id < sched->n_backends - 1 &&
        ggml_backend_sched_is_cuda_host_buffer(buf) &&
        ggml_backend_sched_is_staged_kv_input(t, consumer)) {
        return false;
    }

    return buft != NULL && ggml_backend_supports_buft(sched->backends[backend_id], buft);
}

static void ggml_backend_sched_set_if_supported(ggml_backend_sched_t sched, struct ggml_tensor * node, int cur_backend_id, int * node_backend_id) {
    if (ggml_backend_supports_op(sched->backends[cur_backend_id], node)) {
        *node_backend_id = cur_backend_id;
        SET_CAUSE(node, "2.sup");
    }
}

// assigns backends to ops and splits the graph into subgraphs that can be computed on the same backend
void ggml_backend_sched_split_graph(ggml_backend_sched_t sched, struct ggml_cgraph * graph) {
    // reset splits
    sched->n_splits = 0;
    sched->n_graph_inputs = 0;
    sched->is_reset = false;

    struct ggml_init_params params = {
        /* .mem_size =   */ sched->context_buffer_size,
        /* .mem_buffer = */ sched->context_buffer,
        /* .no_alloc =   */ true
    };

    ggml_free(sched->ctx);

    sched->ctx = ggml_init(params);
    if (sched->ctx == NULL) {
        GGML_ABORT("%s: failed to initialize context\n", __func__);
    }
    memset(sched->staging_roots, 0, sizeof(sched->staging_roots));
    memset(sched->weight_cache_roots, 0, sizeof(sched->weight_cache_roots));

    bool graph_prefill = false;
    bool graph_has_target_gate = false;
    for (int i = 0; i < graph->n_nodes; ++i) {
        const struct ggml_tensor * node = graph->nodes[i];
        for (int j = 0; j < GGML_MAX_SRC; ++j) {
            const struct ggml_tensor * src = node->src[j];
            if (src != NULL && strcmp(src->name, "blk.0.ffn_gate.weight") == 0) {
                graph_has_target_gate = true;
                // Attention score tensors grow with KV length even for a
                // one-token decode, so scanning arbitrary graph dimensions
                // misclassifies long-context decode as prefill. The target
                // gate matmul's token dimension is the actual graph batch.
                const int64_t gate_batch = ggml_backend_sched_op_batch_size(node);
                graph_prefill = gate_batch >= 32;
                if (getenv("LLAMA_WEIGHT_CACHE_TRACE") != NULL) {
                    fprintf(stderr,
                            "WEIGHT_CACHE_GRAPH_GATE: node=%s op=%s ne=%lld,%lld,%lld,%lld batch=%lld\n",
                            node->name, ggml_op_name(node->op),
                            (long long) node->ne[0], (long long) node->ne[1],
                            (long long) node->ne[2], (long long) node->ne[3],
                            (long long) gate_batch);
                    fflush(stderr);
                }
            }
            if (src != NULL && strcmp(src->name, "output.weight") == 0) {
                // Model weights have process lifetime and outlive scheduler
                // graph contexts, so retaining this pointer is safe.
                sched->weight_cache_alias_output = const_cast<struct ggml_tensor *>(src);
            }
        }
    }
    if (graph_prefill && graph_has_target_gate) {
        // Only the target context sees the full backbone during prompt eval.
        // The MTP draft context must not allocate a second 1 GiB cache.
        sched->weight_cache_owner = true;
    }

    const bool alias_output = ggml_backend_sched_weight_cache_alias_output_enabled();
    if (alias_output && sched->weight_cache_owner && sched->weight_cache_alias_output != NULL) {
        // Decode graphs need the original output matrix from their start. A
        // prefill graph restores it later, immediately before the split which
        // consumes output.weight, after the cached early-layer gates are done.
        if (sched->weight_cache_alias_dirty && !graph_prefill) {
            const bool alias_had_entries = sched->weight_cache_n > 0;
            ggml_backend_tensor_set(sched->weight_cache_alias_output,
                    sched->weight_cache_alias_shadow, 0, sched->weight_cache_alias_size);
            sched->weight_cache_alias_dirty = false;
            sched->weight_cache_alias_completed = sched->weight_cache_alias_completed || alias_had_entries;
            sched->weight_cache_n = 0;
            sched->weight_cache_used = 0;
            if (getenv("LLAMA_WEIGHT_CACHE_TRACE") != NULL) {
                fprintf(stderr, "WEIGHT_CACHE_ALIAS_RESTORE: output.weight bytes=%zu\n",
                        sched->weight_cache_alias_size);
                fflush(stderr);
            }
        }

        // The alias is a per-prefill scratch cache, not a one-shot process
        // resource.  Server warmup also builds a prefill-shaped graph and used
        // to set alias_completed permanently, which disabled the cache before
        // the first real request.  A completed prefill always restores
        // output.weight and clears the entries below, so it is safe to arm the
        // alias again for the next genuine prefill.
        sched->weight_cache_alias_can_fill = graph_prefill;
        if (sched->weight_cache_alias_can_fill && sched->weight_cache_alias_shadow == NULL) {
            sched->weight_cache_alias_size = ggml_nbytes(sched->weight_cache_alias_output);
            sched->weight_cache_alias_shadow = malloc(sched->weight_cache_alias_size);
            GGML_ASSERT(sched->weight_cache_alias_shadow != NULL);
            ggml_backend_tensor_get(sched->weight_cache_alias_output,
                    sched->weight_cache_alias_shadow, 0, sched->weight_cache_alias_size);
            if (getenv("LLAMA_WEIGHT_CACHE_TRACE") != NULL) {
                fprintf(stderr, "WEIGHT_CACHE_ALIAS_SHADOW: output.weight bytes=%zu\n",
                        sched->weight_cache_alias_size);
                fflush(stderr);
            }
        }
    } else {
        sched->weight_cache_alias_can_fill = false;
    }
    if (sched->weight_cache_owner &&
        (!sched->weight_cache_phase_set || sched->weight_cache_prefill != graph_prefill)) {
        sched->weight_cache_prefill = graph_prefill;
        sched->weight_cache_phase_set = true;
        // Alias entries remain valid across consecutive prompt chunks. They are
        // cleared only when output.weight is restored. The ordinary dedicated
        // cache retains the original phase-reset behavior.
        if (!alias_output || !sched->weight_cache_alias_dirty) {
            sched->weight_cache_n = 0;
            sched->weight_cache_used = 0;
        }
        if (getenv("LLAMA_WEIGHT_CACHE_TRACE") != NULL) {
            fprintf(stderr, "WEIGHT_CACHE_PHASE: %s\n", graph_prefill ? "prefill" : "decode");
            fflush(stderr);
        }
    }

    graph->uid = ggml_graph_next_uid();

    // pass 1: assign backends to ops with pre-allocated inputs
    for (int i = 0; i < graph->n_leafs; i++) {
        struct ggml_tensor * leaf = graph->leafs[i];
        int * leaf_backend_id = &tensor_backend_id(leaf);
        // do not overwrite user assignments
        if (*leaf_backend_id == -1) {
            *leaf_backend_id = ggml_backend_sched_backend_id_from_cur(sched, leaf);
        }
    }

    for (int i = 0; i < graph->n_nodes; i++) {
        struct ggml_tensor * node = graph->nodes[i];
        int * node_backend_id = &tensor_backend_id(node);
        // do not overwrite user assignments
        if (*node_backend_id == -1) {
            *node_backend_id = ggml_backend_sched_backend_id_from_cur(sched, node);

#if 0
            // src
            if (node->op == GGML_OP_NONE) {
                continue;
            }

            for (int j = 0; j < GGML_MAX_SRC; j++) {
                struct ggml_tensor * src = node->src[j];
                if (src == NULL) {
                    continue;
                }
                int * src_backend_id = &tensor_backend_id(src);
                if (*src_backend_id == -1) {
                    *src_backend_id = ggml_backend_sched_backend_id_from_cur(sched, src);
                }
            }
#endif
        }
    }

    // pass 2: expand current backend assignments
    // assign the same backend to adjacent nodes
    // expand gpu backends (i.e. non last prio) up and down, ignoring cpu (the lowest priority backend)
    // thus, cpu will never be used unless weights are on cpu, or there are no gpu ops between cpu ops
    // ops unsupported by the backend being expanded will be left unassigned so that they can be assigned later when the locations of its inputs are known
    // expand gpu down
    {
        int cur_backend_id = -1;
        for (int i = 0; i < graph->n_nodes; i++) {
            struct ggml_tensor * node = graph->nodes[i];
            if (ggml_is_view_op(node->op)) {
                continue;
            }
            int * node_backend_id = &tensor_backend_id(node);
            if (*node_backend_id != -1) {
                if (*node_backend_id == sched->n_backends - 1) {
                    // skip cpu (lowest prio backend)
                    cur_backend_id = -1;
                } else {
                    cur_backend_id = *node_backend_id;
                }
            } else if (cur_backend_id != -1) {
                ggml_backend_sched_set_if_supported(sched, node, cur_backend_id, node_backend_id);
            }
        }
    }
    // expand gpu up
    {
        int cur_backend_id = -1;
        for (int i = graph->n_nodes - 1; i >= 0; i--) {
            struct ggml_tensor * node = graph->nodes[i];
            if (ggml_is_view_op(node->op)) {
                continue;
            }
            int * node_backend_id = &tensor_backend_id(node);
            if (*node_backend_id != -1) {
                if (*node_backend_id == sched->n_backends - 1) {
                    // skip cpu (lowest prio backend)
                    cur_backend_id = -1;
                } else {
                    cur_backend_id = *node_backend_id;
                }
            } else if (cur_backend_id != -1) {
                ggml_backend_sched_set_if_supported(sched, node, cur_backend_id, node_backend_id);
            }
        }
    }
    // expand rest down
    {
        int cur_backend_id = -1;
        for (int i = 0; i < graph->n_nodes; i++) {
            struct ggml_tensor * node = graph->nodes[i];
            if (ggml_is_view_op(node->op)) {
                continue;
            }
            int * node_backend_id = &tensor_backend_id(node);
            if (*node_backend_id != -1) {
                cur_backend_id = *node_backend_id;
            } else if (cur_backend_id != -1) {
                ggml_backend_sched_set_if_supported(sched, node, cur_backend_id, node_backend_id);
            }
        }
    }
    // expand rest up
    {
        int cur_backend_id = -1;
        for (int i = graph->n_nodes - 1; i >= 0; i--) {
            struct ggml_tensor * node = graph->nodes[i];
            if (ggml_is_view_op(node->op)) {
                continue;
            }
            int * node_backend_id = &tensor_backend_id(node);
            if (*node_backend_id != -1) {
                cur_backend_id = *node_backend_id;
            } else if (cur_backend_id != -1) {
                ggml_backend_sched_set_if_supported(sched, node, cur_backend_id, node_backend_id);
            }
        }
    }

    // pass 3: upgrade nodes to higher prio backends with compatible buffer types
    // if the tensor is already in the same buffer type (*) as another higher priority backend, we should move it there
    // however, we also need to verify that the sources are in compatible buffer types
    // (*) the actual requirement is more relaxed, the buffer type of the backend should be supported by all the users of this tensor further down the graph
    // however, this is slow to verify, so we have a more strict requirement that the buffer type is the same
    // this is not uncommon since multiple backends can use host memory, with the same buffer type (eg. BLAS and CPU)
    // additionally, set remaining unassigned nodes to the backend with the most supported inputs
    // only nodes that could not be assigned during expansion due to the backend not supporting the op should be unassigned at this point
    for (int i = 0; i < graph->n_nodes; i++) {
        struct ggml_tensor * node = graph->nodes[i];
        if (ggml_is_view_op(node->op)) {
            continue;
        }
        int * node_backend_id = &tensor_backend_id(node);
        if (*node_backend_id == -1) {
            // unassigned node: find the backend with the most supported inputs
            int n_supported_best = -1;
            for (int b = 0; b < sched->n_backends; b++) {
                if (ggml_backend_supports_op(sched->backends[b], node)) {
                    int n_supported = 0;
                    for (int j = 0; j < GGML_MAX_SRC; j++) {
                        struct ggml_tensor * src = node->src[j];
                        if (src == NULL) {
                            continue;
                        }
                        if ((tensor_backend_id(src) != -1 || tensor_backend_id(src->view_src) != -1) && ggml_backend_sched_buffer_supported(sched, src, b, node)) {
                            n_supported++;
                        }
                    }
                    if (n_supported > n_supported_best) {
                        n_supported_best = n_supported;
                        *node_backend_id = b;
                        SET_CAUSE(node, "3.best");
                    }
                }
            }
        } else {
            // assigned node: upgrade to higher prio backend if possible
            for (int b = 0; b < *node_backend_id; b++) {
                if (sched->bufts[b] == sched->bufts[*node_backend_id] && ggml_backend_supports_op(sched->backends[b], node)) {
                    bool supported = true;
                    for (int j = 0; j < GGML_MAX_SRC; j++) {
                        struct ggml_tensor * src = node->src[j];
                        if (src == NULL) {
                            continue;
                        }
                        if (!ggml_backend_sched_buffer_supported(sched, src, b, node)) {
                            supported = false;
                            break;
                        }
                    }
                    if (supported) {
                        *node_backend_id = b;
                        SET_CAUSE(node, "3.upg");
                        break;
                    }
                }
            }
        }
    }

    // pass 4: assign backends to remaining src from dst and view_src
    for (int i = 0; i < graph->n_nodes; i++) {
        struct ggml_tensor * node = graph->nodes[i];
        int * cur_backend_id = &tensor_backend_id(node);
        if (node->view_src != NULL && *cur_backend_id == -1) {
            *cur_backend_id = tensor_backend_id(node->view_src);
            SET_CAUSE(node, "4.vsrc");
        }
        for (int j = 0; j < GGML_MAX_SRC; j++) {
            struct ggml_tensor * src = node->src[j];
            if (src == NULL) {
                continue;
            }
            int * src_backend_id = &tensor_backend_id(src);
            if (*src_backend_id == -1) {
                if (src->view_src != NULL) {
                    // views are always on the same backend as the source
                    *src_backend_id = tensor_backend_id(src->view_src);
                    SET_CAUSE(src, "4.vsrc");
                } else {
                    *src_backend_id = *cur_backend_id;
                    SET_CAUSE(src, "4.cur");
                }
            }
        }
        // if the node is still unassigned, assign it to the first backend that supports it
        for (int b = 0; b < sched->n_backends && *cur_backend_id == -1; b++) {
            ggml_backend_sched_set_if_supported(sched, node, b, cur_backend_id);
        }
        GGML_ASSERT(*cur_backend_id != -1);

        // Small decode matmuls backed by CUDA_Host weights normally remain on
        // the CPU. Promote only the leading gates that fit completely in the
        // persistent cache; the remaining gates keep the proven CPU path and
        // therefore do not trigger per-token PCIe copies.
        if (sched->weight_cache_owner &&
            !sched->weight_cache_prefill &&
            ggml_backend_sched_weight_cache_decode_gates_enabled()) {
            for (int j = 0; j < GGML_MAX_SRC; ++j) {
                const struct ggml_tensor * src = node->src[j];
                if (!ggml_backend_sched_weight_cache_decode_gate_fits(src)) {
                    continue;
                }
                for (int b = 0; b < sched->n_backends - 1; ++b) {
                    if (ggml_backend_supports_op(sched->backends[b], node)) {
                        *cur_backend_id = b;
                        break;
                    }
                }
                break;
            }
        }

        if (getenv("LLAMA_WEIGHT_CACHE_TRACE") != NULL) {
            for (int j = 0; j < GGML_MAX_SRC; ++j) {
                struct ggml_tensor * src = node->src[j];
                if (src != NULL && strcmp(src->name, "blk.0.ffn_gate.weight") == 0) {
                    fprintf(stderr,
                            "WEIGHT_CACHE_ASSIGN: phase=%s node_backend=%s src_backend=%s\n",
                            sched->weight_cache_prefill ? "prefill" : "decode",
                            ggml_backend_name(sched->backends[*cur_backend_id]),
                            ggml_backend_name(sched->backends[tensor_backend_id(src)]));
                    fflush(stderr);
                }
            }
        }
    }

    // pass 5: split graph, find tensors that need to be copied
    {
        int i_split = 0;
        struct ggml_backend_sched_split * split = &sched->splits[0];
        // find the backend of the first split, skipping view ops
        int i = 0;
        for (; i < graph->n_nodes; i++) {
            struct ggml_tensor * node = graph->nodes[i];
            if (!ggml_is_view_op(node->op)) {
                split->backend_id = tensor_backend_id(node);
                break;
            }
        }
        split->i_start = 0;
        split->n_inputs = 0;
        split->copy_id = sched->cur_copy;
        split->staging_id = -1;
        split->staging_used = 0;
        int next_staging_id = 0;
        int cur_backend_id = split->backend_id;
        for (; i < graph->n_nodes; i++) {
            struct ggml_tensor * node = graph->nodes[i];

            if (ggml_is_view_op(node->op)) {
                continue;
            }

            const int node_backend_id = tensor_backend_id(node);

            GGML_ASSERT(node_backend_id != -1); // all nodes should be assigned by now, this can happen if there is no CPU fallback

            // check if we should start a new split based on the sources of the current node
            bool need_new_split = false;
            if (node_backend_id == cur_backend_id && split->n_inputs > 0) {
                for (int j = 0; j < GGML_MAX_SRC; j++) {
                    struct ggml_tensor * src = node->src[j];
                    if (src == NULL) {
                        continue;
                    }
                    // check if a weight is on a different and incompatible backend
                    // by starting a new split, the memory of the previously offloaded weights can be reused
                    if (src->buffer != NULL && src->buffer->usage == GGML_BACKEND_BUFFER_USAGE_WEIGHTS) {
                        int src_backend_id = tensor_backend_id(src);
                        const struct ggml_backend_sched_weight_cache_entry * cache_entry =
                                ggml_backend_sched_weight_cache_find(sched, src);
                        const bool cached_on_target = cache_entry != NULL && cache_entry->valid;
                        if (src_backend_id != cur_backend_id &&
                            !cached_on_target &&
                            !ggml_backend_sched_buffer_supported(sched, src, cur_backend_id, node)) {
                            need_new_split = true;
                            break;
                        }
                    }
                    // check if the split has too many inputs
                    // FIXME: count the number of inputs instead of only checking when full
                    if (split->n_inputs >= split->inputs_capacity) {
                        const size_t id = hash_id(src);
                        int src_backend_id = sched->hv_tensor_backend_ids[id];
                        bool supported = ggml_backend_sched_buffer_supported(sched, src, cur_backend_id, node);
                        if (src_backend_id != cur_backend_id && tensor_id_copy(id, cur_backend_id, 0) == NULL && !supported) {
                            need_new_split = true;
                            break;
                        }
                    }
                }
            }

            if (node_backend_id != cur_backend_id || need_new_split) {
                split->i_end = i;
                i_split++;
                if (i_split >= sched->splits_capacity) {
                    int old_cap = sched->splits_capacity;
                    sched->splits_capacity *= 2;
                    sched->splits = (ggml_backend_sched_split *)
                        realloc(sched->splits, sched->splits_capacity * sizeof(struct ggml_backend_sched_split));
                    GGML_ASSERT(sched->splits != NULL);
                    for (int k = old_cap; k < sched->splits_capacity; k++) {
                        memset(&sched->splits[k], 0, sizeof(struct ggml_backend_sched_split));
                    }
                }
                split = &sched->splits[i_split];
                split->backend_id = node_backend_id;
                split->copy_id = sched->cur_copy;
                split->staging_id = -1;
                split->staging_used = 0;
                split->i_start = i;
                split->n_inputs = 0;
                cur_backend_id = node_backend_id;
            }

            // find inputs that are not on the same backend
            for (int j = 0; j < GGML_MAX_SRC; j++) {
                struct ggml_tensor * src = node->src[j];
                if (src == NULL) {
                    continue;
                }

                size_t src_id = hash_id(src);
                const int src_backend_id = sched->hv_tensor_backend_ids[src_id];
                GGML_ASSERT(src_backend_id != -1); // all inputs should be assigned by now

                if (src->flags & GGML_TENSOR_FLAG_INPUT && sched->n_copies > 1) {
                    if (tensor_id_copy(src_id, src_backend_id, 0) == NULL) {
                        ggml_backend_t backend = sched->backends[src_backend_id];
                        for (int c = 0; c < sched->n_copies; c++) {
                            struct ggml_tensor * tensor_copy;
                            if (c == sched->cur_copy) {
                                tensor_copy = src; // use the original tensor as the current copy
                            } else {
                                tensor_copy = ggml_dup_tensor_layout(sched->ctx, src);
                                ggml_format_name(tensor_copy, "%s#%s#%d", ggml_backend_name(backend), src->name, c);
                            }
                            ggml_set_input(tensor_copy);
                            ggml_set_output(tensor_copy); // prevent ggml-alloc from overwriting the tensor
                            tensor_id_copy(src_id, src_backend_id, c) = tensor_copy;
                            SET_CAUSE(tensor_copy, "4.cpy");
                        }
                        int n_graph_inputs = sched->n_graph_inputs++;
                        if (n_graph_inputs >= sched->graph_inputs_capacity) {
                            ggml_backend_sched_graph_inputs_grow(sched);
                        }
                        sched->graph_inputs[n_graph_inputs] = src;
                    }
                }

                if (src_backend_id != cur_backend_id && !ggml_backend_sched_buffer_supported(sched, src, cur_backend_id, node)) {
                    // create a copy of the input in the split's backend
                    if (tensor_id_copy(src_id, cur_backend_id, 0) == NULL) {
                        ggml_backend_t backend = sched->backends[cur_backend_id];
                        struct ggml_backend_sched_weight_cache_entry * weight_cache_entry = NULL;
                        if (cur_backend_id < sched->n_backends - 1 &&
                            ggml_backend_sched_weight_cache_name_eligible(sched, src)) {
                            weight_cache_entry = ggml_backend_sched_weight_cache_get_or_add(sched, src);
                        }
                        const bool staged_weight = sched->staging_double_buffer &&
                                weight_cache_entry == NULL &&
                                src->buffer != NULL &&
                                src->buffer->usage == GGML_BACKEND_BUFFER_USAGE_WEIGHTS &&
                                ggml_backend_sched_is_cuda_host_buffer(src->buffer) &&
                                cur_backend_id < sched->n_backends - 1;
                        const bool staged_kv = sched->staging_double_buffer &&
                                ggml_backend_sched_is_cuda_host_buffer(src->view_src ? src->view_src->buffer : src->buffer) &&
                                ggml_backend_sched_is_staged_kv_input(src, node) &&
                                cur_backend_id < sched->n_backends - 1;
                        const bool staged_input = staged_weight || staged_kv;
                        if (staged_input && split->staging_id < 0) {
                            split->staging_id = next_staging_id;
                            next_staging_id = (next_staging_id + 1) % sched->staging_slots;
                        }
                        for (int c = 0; c < sched->n_copies; c++) {
                            struct ggml_tensor * tensor_copy = ggml_dup_tensor_layout(sched->ctx, src);
                            ggml_format_name(tensor_copy, "%s#%s#%d", ggml_backend_name(backend), src->name, c);
                            const bool staging_galloc = getenv("LLAMA_STAGING_GALLOC") != NULL;
                            if (weight_cache_entry != NULL) {
                                if (ggml_backend_sched_weight_cache_alias_output_enabled() &&
                                    sched->weight_cache_alias_output != NULL) {
                                    tensor_copy->view_src = sched->weight_cache_alias_output;
                                } else {
                                    struct ggml_tensor * root = ggml_backend_sched_weight_cache_root(sched, cur_backend_id);
                                    tensor_copy->view_src = root;
                                }
                                tensor_copy->view_offs = weight_cache_entry->offset;
                            } else if (staged_input && !staging_galloc) {
                                struct ggml_tensor * root = ggml_backend_sched_staging_root(sched, cur_backend_id, split->staging_id);
                                const size_t alignment = 256;
                                const size_t offset = (split->staging_used + alignment - 1) & ~(alignment - 1);
                                GGML_ASSERT(offset + ggml_nbytes(tensor_copy) <= ggml_nbytes(root));
                                tensor_copy->view_src = root;
                                tensor_copy->view_offs = offset;
                                split->staging_used = offset + ggml_nbytes(tensor_copy);
                            } else if (!staged_input && sched->n_copies > 1) {
                                ggml_set_input(tensor_copy);
                                ggml_set_output(tensor_copy); // prevent ggml-alloc from overwriting the tensor
                            }
                            tensor_id_copy(src_id, cur_backend_id, c) = tensor_copy;
                            SET_CAUSE(tensor_copy, "4.cpy");
                        }
                        int n_inputs = split->n_inputs++;
                        if (n_inputs >= split->inputs_capacity) {
                            ggml_backend_sched_split_inputs_grow(split);
                        }
                        split->inputs[n_inputs] = src;
                    }
                    node->src[j] = tensor_id_copy(src_id, cur_backend_id, split->copy_id);
                }
            }
        }
        split->i_end = graph->n_nodes;
        sched->n_splits = i_split + 1;
    }

    if (sched->debug) {
        ggml_backend_sched_print_assignments(sched, graph);
    }

    // swap node_backend_ids and leaf _backend_ids with prevs
    {
        int * tmp = sched->node_backend_ids;
        sched->node_backend_ids = sched->prev_node_backend_ids;
        sched->prev_node_backend_ids = tmp;

        tmp = sched->leaf_backend_ids;
        sched->leaf_backend_ids = sched->prev_leaf_backend_ids;
        sched->prev_leaf_backend_ids = tmp;
    }

    int total_inputs = sched->n_graph_inputs;
    for (int i = 0; i < sched->n_splits; i++) {
        total_inputs += sched->splits[i].n_inputs;
    }
    int graph_size = std::max(graph->n_nodes, graph->n_leafs) + total_inputs * 2 * sched->n_copies + sched->n_backends * 3;

    // remember the actual graph_size for performing reallocation checks later [GGML_SCHED_DEBUG_REALLOC]
    sched->debug_prev_graph_size = sched->debug_graph_size;
    sched->debug_graph_size = graph_size;

    if (sched->graph.size < graph_size) {
        sched->graph.size = graph_size;
        sched->graph.nodes = (ggml_tensor **) realloc(sched->graph.nodes, graph_size * sizeof(struct ggml_tensor *));
        sched->graph.leafs = (ggml_tensor **) realloc(sched->graph.leafs, graph_size * sizeof(struct ggml_tensor *));
        GGML_ASSERT(sched->graph.nodes != NULL);
        GGML_ASSERT(sched->graph.leafs != NULL);
    }
    sched->graph.n_nodes = 0;
    sched->graph.n_leafs = 0;

    struct ggml_cgraph * graph_copy = &sched->graph;

    for (int i = 0; i < sched->n_splits; i++) {
        struct ggml_backend_sched_split * split = &sched->splits[i];
        split->graph = ggml_graph_view(graph, split->i_start, split->i_end);

        // Optimize this split of the graph. This needs to happen before we make graph_copy,
        // so they are in sync.
        ggml_backend_graph_optimize(sched->backends[split->backend_id], &split->graph);

        // add inputs to the graph copy so that they are allocated by ggml-alloc at the start of the split
        for (int j = 0; j < split->n_inputs; j++) {
            assert(graph_copy->size > (graph_copy->n_nodes + 1));

            struct ggml_tensor * input = split->inputs[j];
            const size_t input_id = hash_id(input);
            struct ggml_tensor * input_cpy = tensor_id_copy(input_id, split->backend_id, split->copy_id);

            // add a dependency to the input source so that it is not freed before the copy is done
            struct ggml_tensor * input_dep = ggml_view_tensor(sched->ctx, input);
            input_dep->src[0] = input;
            sched->node_backend_ids[graph_copy->n_nodes] = sched->hv_tensor_backend_ids[input_id];
            graph_copy->nodes[graph_copy->n_nodes++] = input_dep;

            // add a dependency to the input copy so that it is allocated at the start of the split
            sched->node_backend_ids[graph_copy->n_nodes] = split->backend_id;
            graph_copy->nodes[graph_copy->n_nodes++] = input_cpy;
        }

        for (int j = split->i_start; j < split->i_end; j++) {
            assert(graph_copy->size > graph_copy->n_nodes);
            sched->node_backend_ids[graph_copy->n_nodes] = tensor_backend_id(graph->nodes[j]);
            graph_copy->nodes[graph_copy->n_nodes++] = graph->nodes[j];
        }
    }

    if (sched->n_copies > 1) {
        // add input copies as leafs so that they are allocated first
        for (int i = 0; i < sched->n_graph_inputs; i++) {
            struct ggml_tensor * input = sched->graph_inputs[i];
            size_t id = hash_id(input);
            int backend_id = tensor_backend_id(input);
            for (int c = 0; c < sched->n_copies; c++) {
                struct ggml_tensor * input_cpy = tensor_id_copy(id, backend_id, c);
                sched->leaf_backend_ids[graph_copy->n_leafs] = backend_id;
                assert(graph_copy->size > graph_copy->n_leafs);
                graph_copy->leafs[graph_copy->n_leafs++] = input_cpy;
            }
        }

        for (int i = 0; i < sched->n_splits; i++) {
            struct ggml_backend_sched_split * split = &sched->splits[i];
            int backend_id = split->backend_id;
            for (int j = 0; j < split->n_inputs; j++) {
                struct ggml_tensor * input = split->inputs[j];
                size_t id = hash_id(input);
                for (int c = 0; c < sched->n_copies; c++) {
                    struct ggml_tensor * input_cpy = tensor_id_copy(id, backend_id, c);
                    sched->leaf_backend_ids[graph_copy->n_leafs] = backend_id;
                    assert(graph_copy->size > graph_copy->n_leafs);
                    graph_copy->leafs[graph_copy->n_leafs++] = input_cpy;
                }
            }
        }
    }

    if (sched->staging_double_buffer) {
        for (int b = 0; b < sched->n_backends; ++b) {
            for (int c = 0; c < sched->staging_slots; ++c) {
                struct ggml_tensor * root = sched->staging_roots[b][c];
                if (root == NULL) {
                    continue;
                }
                sched->leaf_backend_ids[graph_copy->n_leafs] = b;
                assert(graph_copy->size > graph_copy->n_leafs);
                graph_copy->leafs[graph_copy->n_leafs++] = root;
            }
        }
    }

    for (int b = 0; b < sched->n_backends; ++b) {
        struct ggml_tensor * root = sched->weight_cache_roots[b];
        if (root == NULL) {
            continue;
        }
        sched->leaf_backend_ids[graph_copy->n_leafs] = b;
        assert(graph_copy->size > graph_copy->n_leafs);
        graph_copy->leafs[graph_copy->n_leafs++] = root;
    }

    // add leafs from the original graph
    for (int i = 0; i < graph->n_leafs; i++) {
        struct ggml_tensor * leaf = graph->leafs[i];
        sched->leaf_backend_ids[graph_copy->n_leafs] = tensor_backend_id(leaf);
        assert(graph_copy->size > graph_copy->n_leafs);
        graph_copy->leafs[graph_copy->n_leafs++] = leaf;
    }

    // set ids for all splits
    for (int i = 0; i < sched->n_splits; ++i) {
        sched->splits[i].graph.uid = ggml_graph_next_uid();
    }
}

static bool ggml_backend_sched_alloc_splits(ggml_backend_sched_t sched) {
    bool backend_ids_changed = false;
    for (int i = 0; i < sched->graph.n_nodes; i++) {
        if (sched->node_backend_ids[i] != sched->prev_node_backend_ids[i] &&
            sched->bufts[sched->node_backend_ids[i]] != sched->bufts[sched->prev_node_backend_ids[i]]) {
            backend_ids_changed = true;
            break;
        }
    }
    if (!backend_ids_changed) {
        for (int i = 0; i < sched->graph.n_leafs; i++) {
            if (sched->leaf_backend_ids[i] != sched->prev_leaf_backend_ids[i] &&
                sched->bufts[sched->leaf_backend_ids[i]] != sched->bufts[sched->prev_leaf_backend_ids[i]]) {
                backend_ids_changed = true;
                break;
            }
        }
    }

    // allocate graph
    if (backend_ids_changed || !ggml_gallocr_alloc_graph(sched->galloc, &sched->graph)) {
#ifndef NDEBUG
        GGML_LOG_DEBUG("%s: failed to allocate graph, reserving (backend_ids_changed = %d)\n", __func__, backend_ids_changed);
#endif

        if (sched->debug_realloc > 0) {
            // we are interested only in situations where the graph was reallocated even though its size remained the same [GGML_SCHED_DEBUG_REALLOC]
            // example: https://github.com/ggml-org/llama.cpp/pull/17143
            const bool unexpected = !backend_ids_changed && sched->debug_prev_graph_size == sched->debug_graph_size;

            if (unexpected || sched->debug_realloc > 1) {
                GGML_ABORT("%s: unexpected graph reallocation (graph size = %d, nodes = %d, leafs = %d), debug_realloc = %d\n", __func__,
                        sched->debug_graph_size, sched->graph.n_nodes, sched->graph.n_leafs, sched->debug_realloc);
            }
        }

        // the re-allocation may cause the split inputs to be moved to a different address
        // synchronize without ggml_backend_sched_synchronize to avoid changing cur_copy
        for (int i = 0; i < sched->n_backends; i++) {
            ggml_backend_synchronize(sched->backends[i]);
        }

        ggml_gallocr_reserve_n(sched->galloc, &sched->graph, sched->node_backend_ids, sched->leaf_backend_ids);
        if (!ggml_gallocr_alloc_graph(sched->galloc, &sched->graph)) {
            GGML_LOG_ERROR("%s: failed to allocate graph\n", __func__);
            return false;
        }
    }

    return true;
}

static void * ggml_backend_sched_gate_ram_mirror(
        ggml_backend_sched_t sched,
        struct ggml_tensor * src) {
    for (int i = 0; i < sched->gate_ram_mirror_n; ++i) {
        struct ggml_backend_sched_gate_ram_mirror_entry * entry = &sched->gate_ram_mirrors[i];
        if (entry->src == src) {
            return entry->data;
        }
    }

    GGML_ASSERT(sched->gate_ram_mirror_n < sched->gate_ram_mirror_capacity);
    struct ggml_backend_sched_gate_ram_mirror_entry * entry =
            &sched->gate_ram_mirrors[sched->gate_ram_mirror_n++];
    entry->src = src;
    entry->size = ggml_nbytes(src);
    entry->data = ggml_aligned_malloc(entry->size);
    if (entry->data == nullptr) {
        --sched->gate_ram_mirror_n;
        return nullptr;
    }
    memcpy(entry->data, src->data, entry->size);

    if (getenv("LLAMA_CPU_GATE_RAM_MIRROR_TRACE") != nullptr) {
        fprintf(stderr, "CPU_GATE_RAM_MIRROR: src=%s bytes=%zu count=%d\n",
                src->name, ggml_nbytes(src), sched->gate_ram_mirror_n);
        fflush(stderr);
    }
    return entry->data;
}

static const struct ggml_tensor * ggml_backend_sched_gate_wc_mirror(
        ggml_backend_sched_t sched,
        ggml_backend_t gpu_backend,
        const struct ggml_tensor * src,
        int64_t n_rows) {
    if (getenv("LLAMA_STAGING_WC_MIRROR") == nullptr || src == nullptr || n_rows <= 0) {
        return nullptr;
    }
    for (int i = 0; i < sched->gate_wc_mirror_n; ++i) {
        struct ggml_backend_sched_gate_wc_mirror_entry * entry = &sched->gate_wc_mirrors[i];
        if (entry->src == src && entry->n_rows == n_rows) {
            return &entry->alias;
        }
    }

    if (sched->gate_wc_mirror_n >= sched->gate_wc_mirror_capacity) {
        return nullptr;
    }

    struct ggml_tensor prefix = *src;
    prefix.ne[1] = n_rows;
    prefix.ne[2] = 1;
    prefix.ne[3] = 1;
    const size_t bytes = ggml_nbytes(&prefix);

    ggml_backend_buffer_type_t host_buft =
            ggml_backend_dev_host_buffer_type(ggml_backend_get_device(gpu_backend));
    if (host_buft == nullptr) {
        return nullptr;
    }
    ggml_backend_buffer_t buffer = ggml_backend_buft_alloc_buffer(host_buft, bytes);
    if (buffer == nullptr) {
        return nullptr;
    }

    struct ggml_backend_sched_gate_wc_mirror_entry * entry =
            &sched->gate_wc_mirrors[sched->gate_wc_mirror_n];
    entry->src = src;
    entry->n_rows = n_rows;
    entry->buffer = buffer;
    entry->alias = prefix;
    entry->alias.buffer = buffer;
    entry->alias.view_src = nullptr;
    entry->alias.view_offs = 0;
    entry->alias.data = ggml_backend_buffer_get_base(buffer);

    // Copy once from the loader's ordinary mapped host allocation into a
    // write-combined pinned prefix. The CPU keeps reading the untouched tail
    // from the original tensor, while repeated H2D transfers use the mirror.
    ggml_backend_tensor_get(src, entry->alias.data, 0, bytes);
    ++sched->gate_wc_mirror_n;

    if (getenv("LLAMA_STAGING_WC_MIRROR_TRACE") != nullptr) {
        fprintf(stderr, "GATE_WC_MIRROR: src=%s rows=%lld bytes=%zu count=%d\n",
                src->name, (long long) n_rows, bytes, sched->gate_wc_mirror_n);
        fflush(stderr);
    }
    return &entry->alias;
}

static enum ggml_status ggml_backend_sched_compute_gate_row_split(
        ggml_backend_sched_t sched,
        struct ggml_backend_sched_split * split,
        int cpu_backend_id,
        int weight_src_id,
        int gpu_percent,
        struct ggml_backend_sched_gate_row_split_pending * pending,
        struct ggml_backend_sched_gate_row_split_prefetch * prefetch) {
    struct ggml_tensor * node = split->graph.nodes[0];
    struct ggml_tensor * weight = node->src[weight_src_id];
    struct ggml_tensor * input_cpu = node->src[weight_src_id == 0 ? 1 : 0];

    if (node->op != GGML_OP_MUL_MAT || weight->ne[2] != 1 || weight->ne[3] != 1 || input_cpu == nullptr ||
            node->type != GGML_TYPE_F32 || input_cpu->type != GGML_TYPE_F32) {
        return GGML_STATUS_FAILED;
    }

    int gpu_backend_id = -1;
    for (int backend_id = 0; backend_id < sched->n_backends; ++backend_id) {
        if (ggml_backend_dev_type(ggml_backend_get_device(sched->backends[backend_id])) ==
                GGML_BACKEND_DEVICE_TYPE_GPU) {
            gpu_backend_id = backend_id;
            break;
        }
    }
    if (gpu_backend_id < 0) {
        return GGML_STATUS_FAILED;
    }

    struct ggml_tensor * input_gpu = nullptr;
    for (int input_id = 0; input_id < split->n_inputs; ++input_id) {
        struct ggml_tensor * candidate = split->inputs[input_id];
        if (tensor_copy(candidate, cpu_backend_id, split->copy_id) == input_cpu) {
            input_gpu = candidate;
            break;
        }
    }
    if (input_gpu == nullptr) {
        return GGML_STATUS_FAILED;
    }

    const int64_t n_in = weight->ne[0];
    const int64_t n_out = weight->ne[1];
    const int64_t n_batch = input_cpu->ne[1];
    int64_t n_gpu = (n_out * std::max(5, std::min(95, gpu_percent))) / 100;
    n_gpu = std::max<int64_t>(256, (n_gpu / 256) * 256);
    n_gpu = std::min<int64_t>(n_out - 256, n_gpu);
    const int64_t n_cpu = n_out - n_gpu;
    if (n_cpu <= 0 || n_batch <= 0 || n_batch >= 32) {
        return GGML_STATUS_FAILED;
    }

    ggml_backend_t cpu_backend = sched->backends[cpu_backend_id];
    ggml_backend_t gpu_backend = sched->backends[gpu_backend_id];

    auto ensure_buffer = [](ggml_backend_buffer_t & buffer, size_t & capacity,
                            ggml_backend_t backend, struct ggml_tensor * tensor) {
        ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(backend);
        const size_t needed = ggml_backend_buft_get_alloc_size(buft, tensor);
        if (buffer != nullptr && capacity < needed) {
            ggml_backend_synchronize(backend);
            ggml_backend_buffer_free(buffer);
            buffer = nullptr;
            capacity = 0;
        }
        if (buffer == nullptr) {
            buffer = ggml_backend_alloc_buffer(backend, needed);
            if (buffer == nullptr) {
                return false;
            }
            capacity = ggml_backend_buffer_get_size(buffer);
        }
        return ggml_backend_tensor_alloc(buffer, tensor, ggml_backend_buffer_get_base(buffer)) ==
                GGML_STATUS_SUCCESS;
    };

    struct ggml_backend_sched_gate_row_split_plan * plan = sched->gate_split_plans[n_batch];
    if (plan == nullptr) {
        plan = new (std::nothrow) ggml_backend_sched_gate_row_split_plan {};
        if (plan == nullptr) {
            return GGML_STATUS_ALLOC_FAILED;
        }

        struct ggml_init_params params = {
            /* .mem_size   = */ 1024*1024,
            /* .mem_buffer = */ nullptr,
            /* .no_alloc   = */ true,
        };
        plan->ctx = ggml_init(params);
        if (plan->ctx == nullptr) {
            delete plan;
            return GGML_STATUS_ALLOC_FAILED;
        }

        auto make_alias_2d = [&](const struct ggml_tensor * src, int64_t ne0, int64_t ne1, size_t offset) {
            struct ggml_tensor * alias = ggml_new_tensor_2d(plan->ctx, src->type, ne0, ne1);
            alias->buffer = src->buffer;
            alias->data = (char *) src->data + offset;
            alias->nb[0] = src->nb[0];
            alias->nb[1] = src->nb[1];
            alias->nb[2] = src->nb[2];
            alias->nb[3] = src->nb[3];
            alias->extra = src->extra;
            return alias;
        };

        plan->weight_cpu = make_alias_2d(weight, n_in, n_cpu, (size_t) n_gpu * weight->nb[1]);
        plan->input_cpu  = make_alias_2d(input_cpu, n_in, n_batch, 0);
        plan->output_cpu = ggml_mul_mat(plan->ctx, plan->weight_cpu, plan->input_cpu);
        plan->input_gpu  = make_alias_2d(input_gpu, n_in, n_batch, 0);
        ggml_set_name(plan->output_cpu, "gate_row_split.output_cpu");
        for (int slot = 0; slot < 2; ++slot) {
            plan->weight_gpu[slot] = ggml_new_tensor_2d(plan->ctx, weight->type, n_in, n_gpu);
            plan->output_gpu[slot] = ggml_mul_mat(plan->ctx, plan->weight_gpu[slot], plan->input_gpu);
            ggml_format_name(plan->weight_gpu[slot], "gate_row_split.weight_gpu_%d", slot);
            ggml_format_name(plan->output_gpu[slot], "gate_row_split.output_gpu_%d", slot);
        }

        // Allocate for the largest decode microbatch up front.  This keeps the
        // shared buffers stable if speculative decoding later changes batch
        // size, so already cached plans retain valid output pointers.
        struct ggml_tensor * output_cpu_max = ggml_new_tensor_2d(plan->ctx, GGML_TYPE_F32, n_cpu, 31);
        struct ggml_tensor * output_gpu_max = ggml_new_tensor_2d(plan->ctx, GGML_TYPE_F32, n_gpu, 31);
        if (!ensure_buffer(sched->gate_split_cpu_output_buffer,
                           sched->gate_split_cpu_output_capacity, cpu_backend, output_cpu_max) ||
            !ensure_buffer(sched->gate_split_gpu_weight_buffer[0],
                           sched->gate_split_gpu_weight_capacity[0], gpu_backend, plan->weight_gpu[0]) ||
            !ensure_buffer(sched->gate_split_gpu_weight_buffer[1],
                           sched->gate_split_gpu_weight_capacity[1], gpu_backend, plan->weight_gpu[1]) ||
            !ensure_buffer(sched->gate_split_gpu_output_buffer,
                           sched->gate_split_gpu_output_capacity, gpu_backend, output_gpu_max) ||
            ggml_backend_tensor_alloc(sched->gate_split_cpu_output_buffer, plan->output_cpu,
                    ggml_backend_buffer_get_base(sched->gate_split_cpu_output_buffer)) != GGML_STATUS_SUCCESS ||
            ggml_backend_tensor_alloc(sched->gate_split_gpu_output_buffer, plan->output_gpu[0],
                    ggml_backend_buffer_get_base(sched->gate_split_gpu_output_buffer)) != GGML_STATUS_SUCCESS ||
            ggml_backend_tensor_alloc(sched->gate_split_gpu_output_buffer, plan->output_gpu[1],
                    ggml_backend_buffer_get_base(sched->gate_split_gpu_output_buffer)) != GGML_STATUS_SUCCESS) {
            ggml_free(plan->ctx);
            delete plan;
            return GGML_STATUS_ALLOC_FAILED;
        }

        plan->graph_cpu = ggml_new_graph_custom(plan->ctx, 8, false);
        ggml_build_forward_expand(plan->graph_cpu, plan->output_cpu);
        for (int slot = 0; slot < 2; ++slot) {
            plan->graph_gpu[slot] = ggml_new_graph_custom(plan->ctx, 8, false);
            ggml_build_forward_expand(plan->graph_gpu[slot], plan->output_gpu[slot]);
        }
        plan->n_in = n_in;
        plan->n_out = n_out;
        plan->n_batch = n_batch;
        plan->n_gpu = n_gpu;
        sched->gate_split_plans[n_batch] = plan;
    } else if (plan->n_in != n_in || plan->n_out != n_out || plan->n_gpu != n_gpu) {
        // The cache is intentionally shape-specific.  Qwen's dense FFN layers
        // share a shape; fail closed rather than reusing metadata incorrectly.
        return GGML_STATUS_FAILED;
    }

    auto refresh_alias = [](struct ggml_tensor * alias, const struct ggml_tensor * src, size_t offset) {
        alias->buffer = src->buffer;
        alias->data = (char *) src->data + offset;
        alias->nb[0] = src->nb[0];
        alias->nb[1] = src->nb[1];
        alias->nb[2] = src->nb[2];
        alias->nb[3] = src->nb[3];
        alias->extra = src->extra;
    };
    refresh_alias(plan->weight_cpu, weight, (size_t) n_gpu * weight->nb[1]);
    refresh_alias(plan->input_cpu, input_cpu, 0);
    refresh_alias(plan->input_gpu, input_gpu, 0);

    const size_t gpu_output_bytes = ggml_nbytes(plan->output_gpu[0]);
    if (sched->gate_split_gpu_output_host_capacity < gpu_output_bytes) {
        void * resized = realloc(sched->gate_split_gpu_output_host, gpu_output_bytes);
        if (resized == nullptr) {
            return GGML_STATUS_ALLOC_FAILED;
        }
        sched->gate_split_gpu_output_host = resized;
        sched->gate_split_gpu_output_host_capacity = gpu_output_bytes;
    }

    GGML_ASSERT(pending != nullptr && !pending->active);
    const int64_t begin_us = ggml_time_us();
    sched->cpu_gpu_overlap->submit(cpu_backend, *plan->graph_cpu);

    int slot = 0;
    const bool prefetched = prefetch != nullptr && prefetch->weight == weight &&
            prefetch->n_gpu == n_gpu && prefetch->slot >= 0;
    if (prefetched) {
        slot = prefetch->slot;
        *prefetch = {};
        prefetch->slot = -1;
    } else {
        struct ggml_tensor weight_gpu_src = *weight;
        weight_gpu_src.ne[1] = n_gpu;
        weight_gpu_src.ne[2] = 1;
        weight_gpu_src.ne[3] = 1;
        snprintf(weight_gpu_src.name, sizeof(weight_gpu_src.name), "%s.row_split", weight->name);
        const struct ggml_tensor * weight_gpu_copy_src = &weight_gpu_src;
        if (const struct ggml_tensor * mirror =
                ggml_backend_sched_gate_wc_mirror(sched, gpu_backend, weight, n_gpu)) {
            weight_gpu_copy_src = mirror;
        }
        bool copied_async = gpu_backend->iface.cpy_tensor_async &&
                gpu_backend->iface.cpy_tensor_async(
                        cpu_backend, gpu_backend, weight_gpu_copy_src, plan->weight_gpu[slot]);
        if (!copied_async) {
            ggml_backend_tensor_copy(weight_gpu_copy_src, plan->weight_gpu[slot]);
        }
    }
    pending->active = true;
    pending->plan = plan;
    pending->output = node;
    pending->gpu_backend = gpu_backend;
    pending->gpu_status = ggml_backend_graph_compute_async(gpu_backend, plan->graph_gpu[slot]);
    pending->gpu_percent = gpu_percent;
    pending->n_gpu = n_gpu;
    pending->n_cpu = n_cpu;
    pending->n_batch = n_batch;
    pending->gpu_output_bytes = gpu_output_bytes;
    pending->begin_us = begin_us;
    pending->slot = slot;
    return GGML_STATUS_SUCCESS;
}

static enum ggml_status ggml_backend_sched_finish_gate_row_split(
        ggml_backend_sched_t sched,
        struct ggml_backend_sched_gate_row_split_pending * pending,
        struct ggml_tensor * output_gpu_dst,
        bool * direct_gpu_ready) {
    GGML_ASSERT(pending != nullptr && pending->active);
    if (direct_gpu_ready != nullptr) {
        *direct_gpu_ready = false;
    }
    const bool direct_gpu = output_gpu_dst != nullptr &&
            getenv("LLAMA_CPU_GPU_GATE_DIRECT_GPU") != nullptr &&
            output_gpu_dst->type == GGML_TYPE_F32 &&
            output_gpu_dst->ne[0] == pending->n_gpu + pending->n_cpu &&
            output_gpu_dst->ne[1] == pending->n_batch;
    const enum ggml_status cpu_status = sched->cpu_gpu_overlap->wait();

    if (pending->gpu_status == GGML_STATUS_SUCCESS && cpu_status == GGML_STATUS_SUCCESS) {
        if (direct_gpu) {
            // The GPU prefix is already on-device.  Copy it directly into the
            // scheduler's imported gate tensor and upload only the CPU tail.
            // Both operations are queued on the same CUDA stream as the
            // following FFN suffix, preserving dependencies without the old
            // GPU->CPU->GPU round trip or changing any numerical operation.
            struct ggml_tensor gpu_prefix_dst = *output_gpu_dst;
            gpu_prefix_dst.ne[0] = pending->n_gpu;
            gpu_prefix_dst.ne[1] = pending->n_batch;
            gpu_prefix_dst.ne[2] = 1;
            gpu_prefix_dst.ne[3] = 1;

            struct ggml_tensor * gpu_prefix_src = pending->plan->output_gpu[pending->slot];
            const bool copied_gpu = pending->gpu_backend->iface.cpy_tensor_async != nullptr &&
                    pending->gpu_backend->iface.cpy_tensor_async(
                            pending->gpu_backend, pending->gpu_backend,
                            gpu_prefix_src, &gpu_prefix_dst);
            if (copied_gpu) {
                ggml_backend_tensor_set_2d_async(
                        pending->gpu_backend, output_gpu_dst,
                        pending->plan->output_cpu->data,
                        (size_t) pending->n_gpu*sizeof(float),
                        (size_t) pending->n_cpu*sizeof(float),
                        (size_t) pending->n_batch,
                        output_gpu_dst->nb[1], pending->plan->output_cpu->nb[1]);
                if (direct_gpu_ready != nullptr) {
                    *direct_gpu_ready = true;
                }
            }
        }

        if (direct_gpu_ready == nullptr || !*direct_gpu_ready) {
            ggml_backend_tensor_get(pending->plan->output_gpu[pending->slot], sched->gate_split_gpu_output_host,
                    0, pending->gpu_output_bytes);
            const float * gpu_data = (const float *) sched->gate_split_gpu_output_host;
            const float * cpu_data = (const float *) pending->plan->output_cpu->data;
            for (int64_t ib = 0; ib < pending->n_batch; ++ib) {
                float * dst = (float *) ((char *) pending->output->data + ib * pending->output->nb[1]);
                memcpy(dst, gpu_data + ib*pending->n_gpu, (size_t) pending->n_gpu*sizeof(float));
                memcpy(dst + pending->n_gpu, cpu_data + ib*pending->n_cpu,
                        (size_t) pending->n_cpu*sizeof(float));
            }
        }
    }

    if (getenv("LLAMA_CPU_GPU_GATE_ROW_SPLIT_TRACE") != nullptr) {
        fprintf(stderr,
                "CPU_GPU_GATE_ROW_SPLIT: gpu_pct=%d rows_gpu=%lld rows_cpu=%lld batch=%lld total_ms=%.3f gpu_status=%d cpu_status=%d direct_gpu=%d\n",
                pending->gpu_percent, (long long) pending->n_gpu, (long long) pending->n_cpu,
                (long long) pending->n_batch, (ggml_time_us() - pending->begin_us)/1000.0,
                (int) pending->gpu_status, (int) cpu_status,
                direct_gpu_ready != nullptr && *direct_gpu_ready ? 1 : 0);
        fflush(stderr);
    }

    const enum ggml_status status = pending->gpu_status != GGML_STATUS_SUCCESS ?
            pending->gpu_status : cpu_status;
    *pending = {};
    return status;
}

static enum ggml_status ggml_backend_sched_compute_splits(ggml_backend_sched_t sched) {
    GGML_ASSERT(sched);
    struct ggml_backend_sched_split * splits = sched->splits;

    ggml_tensor * prev_ids_tensor = nullptr;
    std::vector<int32_t> ids;
    std::vector<ggml_bitset_t> used_ids;

    int prev_backend_id = -1;
    int prev_copy_id = 0;
    const bool cpu_gate_timing = getenv("LLAMA_CPU_GATE_TIMING") != NULL;
    const bool cpu_gpu_ffn_overlap = sched->cpu_gpu_overlap != NULL &&
            getenv("LLAMA_CPU_GPU_FFN_OVERLAP") != NULL && sched->callback_eval == NULL;
    const bool cpu_gate_ram_mirror = getenv("LLAMA_CPU_GATE_RAM_MIRROR") != NULL;
    const int cpu_gpu_gate_row_split = getenv("LLAMA_CPU_GPU_GATE_ROW_SPLIT") != NULL ?
            atoi(getenv("LLAMA_CPU_GPU_GATE_ROW_SPLIT")) : 0;
    const bool cpu_gpu_ffn_overlap_trace = getenv("LLAMA_CPU_GPU_FFN_OVERLAP_TRACE") != NULL;
    int64_t cpu_gate_total_us = 0;
    int cpu_gate_splits = 0;
    struct ggml_tensor * pending_cpu_gate = nullptr;
    struct ggml_tensor * pending_cpu_gate_weight = nullptr;
    void * pending_cpu_gate_weight_data = nullptr;
    int pending_cpu_backend_id = -1;
    int overlap_pairs = 0;
    int overlap_prefix_nodes = 0;
    struct ggml_backend_sched_gate_row_split_pending pending_gate_split = {};
    struct ggml_backend_sched_gate_row_split_prefetch gate_prefetch = {};
    gate_prefetch.slot = -1;

    auto wait_pending_cpu_gate = [&](struct ggml_tensor * output_gpu_dst = nullptr,
                                     bool * direct_gpu_ready = nullptr) -> enum ggml_status {
        if (pending_cpu_gate == nullptr) {
            return GGML_STATUS_SUCCESS;
        }
        const enum ggml_status status = pending_gate_split.active ?
                ggml_backend_sched_finish_gate_row_split(
                        sched, &pending_gate_split, output_gpu_dst, direct_gpu_ready) :
                sched->cpu_gpu_overlap->wait();
        if (pending_cpu_gate_weight != nullptr) {
            pending_cpu_gate_weight->data = pending_cpu_gate_weight_data;
        }
        pending_cpu_gate = nullptr;
        pending_cpu_gate_weight = nullptr;
        pending_cpu_gate_weight_data = nullptr;
        pending_cpu_backend_id = -1;
        return status;
    };

    for (int split_id = 0; split_id < sched->n_splits; split_id++) {
        struct ggml_backend_sched_split * split = &splits[split_id];
        int split_backend_id = split->backend_id;
        ggml_backend_t split_backend = sched->backends[split_backend_id];

        // A pending CPU gate may overlap only with the immediately following
        // GPU split which imports that gate.  If graph splitting ever produces
        // a different shape, conservatively join and use the ordinary path.
        int overlap_gate_input_id = -1;
        if (pending_cpu_gate != nullptr &&
                ggml_backend_dev_type(ggml_backend_get_device(split_backend)) == GGML_BACKEND_DEVICE_TYPE_GPU) {
            for (int input_id = 0; input_id < split->n_inputs; ++input_id) {
                if (split->inputs[input_id] == pending_cpu_gate) {
                    overlap_gate_input_id = input_id;
                    break;
                }
            }
        }
        if (pending_cpu_gate != nullptr && overlap_gate_input_id < 0) {
            const enum ggml_status status = wait_pending_cpu_gate();
            if (status != GGML_STATUS_SUCCESS) {
                return status;
            }
        }

        bool split_uses_alias_output = false;
        if (sched->weight_cache_alias_dirty && sched->weight_cache_alias_output != NULL) {
            for (int node_id = 0; node_id < split->graph.n_nodes && !split_uses_alias_output; ++node_id) {
                const struct ggml_tensor * node = split->graph.nodes[node_id];
                for (int src_id = 0; src_id < GGML_MAX_SRC; ++src_id) {
                    const struct ggml_tensor * src = node->src[src_id];
                    if ((src == sched->weight_cache_alias_output ||
                         (src != NULL && strcmp(src->name, "output.weight") == 0)) &&
                        ggml_backend_sched_op_batch_size(node) > 0) {
                        split_uses_alias_output = true;
                        break;
                    }
                }
            }
        }
        if (split_uses_alias_output) {
            // It is only safe to restore once all alias-backed inputs belong to
            // earlier splits. A staging split which also consumes one would see
            // its gate overwritten by output.weight.
            for (int input_id = 0; input_id < split->n_inputs; ++input_id) {
                struct ggml_backend_sched_weight_cache_entry * entry =
                        ggml_backend_sched_weight_cache_find(sched, split->inputs[input_id]);
                if (entry != NULL && entry->valid) {
                    fprintf(stderr, "WEIGHT_CACHE_ALIAS_UNSAFE_SPLIT: id=%d input=%s\n",
                            split_id, split->inputs[input_id]->name);
                    fflush(stderr);
                    return GGML_STATUS_FAILED;
                }
            }
            const bool alias_had_entries = sched->weight_cache_n > 0;
            ggml_backend_synchronize(split_backend);
            ggml_backend_tensor_set(sched->weight_cache_alias_output,
                    sched->weight_cache_alias_shadow, 0, sched->weight_cache_alias_size);
            sched->weight_cache_alias_dirty = false;
            sched->weight_cache_alias_completed = sched->weight_cache_alias_completed || alias_had_entries;
            sched->weight_cache_n = 0;
            sched->weight_cache_used = 0;
            if (getenv("LLAMA_WEIGHT_CACHE_TRACE") != NULL) {
                fprintf(stderr, "WEIGHT_CACHE_ALIAS_RESTORE_SPLIT: id=%d output.weight bytes=%zu\n",
                        split_id, sched->weight_cache_alias_size);
                fflush(stderr);
            }
        }

        // ensure the previous split's async work has completed before we start
        // this split, the allocator may have reused buffer regions across splits
        if (split->n_inputs == 0 && prev_backend_id >= 0 && prev_backend_id != split_backend_id) {
            if (sched->events[prev_backend_id][prev_copy_id] != NULL) {
                ggml_backend_event_synchronize(sched->events[prev_backend_id][prev_copy_id]);
            } else {
                ggml_backend_synchronize(sched->backends[prev_backend_id]);
            }
        }

        // copy the input tensors to the split backend
        for (int input_id = 0; input_id < split->n_inputs; input_id++) {
            // Defer the tiny CPU gate activation.  The GPU prefix before its
            // first consumer is independent and can execute while the host
            // worker is still producing the activation.
            if (input_id == overlap_gate_input_id) {
                continue;
            }
            ggml_backend_t input_backend = ggml_backend_sched_get_tensor_backend(sched, split->inputs[input_id]);
            struct ggml_tensor * input = split->inputs[input_id];
            struct ggml_tensor * input_cpy = tensor_copy(input, split_backend_id, split->copy_id);
            struct ggml_backend_sched_weight_cache_entry * weight_cache_entry =
                    ggml_backend_sched_weight_cache_find(sched, input);
            const bool weight_cached = weight_cache_entry != NULL &&
                    input_cpy->view_src != NULL &&
                    (strncmp(input_cpy->view_src->name, "weight_cache_root_", 18) == 0 ||
                     input_cpy->view_src == sched->weight_cache_alias_output);
            if (weight_cached && weight_cache_entry->valid) {
                continue;
            }
            const bool staged_input = sched->staging_double_buffer && split->staging_id >= 0 &&
                    input->buffer != NULL &&
                    input->buffer->usage == GGML_BACKEND_BUFFER_USAGE_WEIGHTS &&
                    ggml_backend_sched_is_cuda_host_buffer(input->buffer) &&
                    split_backend_id < sched->n_backends - 1;

            if (input->flags & GGML_TENSOR_FLAG_INPUT) {
                // inputs from the user must be copied immediately to prevent the user overwriting the data before the copy is done
                if (sched->events[split_backend_id][split->copy_id] != NULL) {
                    ggml_backend_event_synchronize(sched->events[split_backend_id][split->copy_id]);
                } else {
                    ggml_backend_synchronize(split_backend);
                }
                ggml_backend_tensor_copy(input, input_cpy);
            } else {
                // wait for the split backend to finish using the input before overwriting it
                if (staged_input) {
                    // A staging slot can be overwritten as soon as the previous
                    // compute that consumed that slot has completed.  The other
                    // slot may still be executing, which is the overlap we want.
                    const int staging_id = split->staging_id;
                    if (sched->staging_event_recorded[split_backend_id][staging_id]) {
                        // The CUDA copy stream can wait on the gate-consumed
                        // event without blocking the scheduler thread. The
                        // legacy host synchronization remains the fallback.
                        if (!ggml_backend_sched_staging_device_reuse_wait_enabled()) {
                            ggml_backend_event_synchronize(sched->staging_events[split_backend_id][staging_id]);
                        }
                    }
                } else if (sched->events[split_backend_id][split->copy_id] != NULL) {
                    if (sched->staging_double_buffer) {
                        ggml_backend_event_synchronize(sched->events[split_backend_id][split->copy_id]);
                    } else {
                        ggml_backend_event_wait(split_backend, sched->events[split_backend_id][split->copy_id]);
                    }
                } else {
                    ggml_backend_synchronize(split_backend);
                }

                // when offloading MoE weights, we can reduce the amount of data copied by copying only the experts that are used
                ggml_tensor * node = split->graph.nodes[0];
                if (split->graph.n_nodes > 0 &&
                    ggml_backend_buffer_get_usage(input->buffer) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS &&
                    ggml_backend_buffer_is_host(input->buffer) && (
                    (node->src[0] == input_cpy && node->op == GGML_OP_MUL_MAT_ID)
                    //|| (node->src[1] == input_cpy && node->op == GGML_OP_ADD_ID) /* GGML_OP_ADD_ID weights are small and not worth splitting */
                    )) {

                    const int64_t n_expert   = node->op == GGML_OP_MUL_MAT_ID ? input->ne[2] : input->ne[1];
                    const size_t expert_size = node->op == GGML_OP_MUL_MAT_ID ? input->nb[2] : input->nb[1];

                    ggml_backend_synchronize(input_backend);

                    // get the ids
                    ggml_tensor * ids_tensor = node->src[2];
                    ggml_backend_t ids_backend = split_backend;

                    // if the ids tensor is also an input of the split, it may not have been copied yet to the split backend
                    // in that case, we use the original ids tensor
                    for (int i = input_id + 1; i < split->n_inputs; i++) {
                        if (ids_tensor == tensor_copy(split->inputs[i], split_backend_id, split->copy_id)) {
                            ids_tensor = split->inputs[i];
                            ids_backend = ggml_backend_sched_get_tensor_backend(sched, split->inputs[i]);
                            break;
                        }
                    }

                    if (ids_tensor != prev_ids_tensor) {
                        ids.resize(ggml_nbytes(ids_tensor) / sizeof(int32_t));
                        ggml_backend_tensor_get_async(ids_backend, ids_tensor, ids.data(), 0, ggml_nbytes(ids_tensor));
                        ggml_backend_synchronize(ids_backend);

                        // find the used experts
                        used_ids.clear();
                        used_ids.resize(ggml_bitset_size(n_expert));
                        for (int64_t i1 = 0; i1 < ids_tensor->ne[1]; i1++) {
                            for (int64_t i0 = 0; i0 < ids_tensor->ne[0]; i0++) {
                                int32_t id = ids[i1 * ids_tensor->nb[1]/sizeof(int32_t) + i0 * ids_tensor->nb[0]/sizeof(int32_t)];
                                GGML_ASSERT(id >= 0 && id < n_expert);
                                ggml_bitset_set(used_ids.data(), id);
                            }
                        }

                        prev_ids_tensor = ids_tensor;
                    }

                    // group consecutive experts and copy them together
                    auto copy_experts = [&](int32_t first_id, int32_t last_id) {
                        const size_t expert_offset = first_id * expert_size;
                        const size_t expert_size_copy =  (last_id - first_id + 1) * expert_size;
                        const size_t padding = std::min<size_t>(expert_size, 512);
                        const size_t padding_end = last_id < n_expert - 1 ? padding : 0;

                        ggml_backend_tensor_set_async(split_backend,
                            input_cpy,
                            (const uint8_t *)input->data + expert_offset, expert_offset,
                            // copy a bit extra at the to ensure there are no NaNs in the padding of the last expert
                            // this is necessary for MMQ in the CUDA backend
                            expert_size_copy + padding_end);
                    };

                    int id = 0;
                    while (!ggml_bitset_get(used_ids.data(), id)) {
                        id++;
                    }
                    int32_t first_id = id;
                    int32_t last_id = first_id;

                    for (++id; id < n_expert; ++id) {
                        if (!ggml_bitset_get(used_ids.data(), id)) {
                            continue;
                        }

                        if (id == last_id + 1) {
                            last_id = id;
                            continue;
                        }

                        copy_experts(first_id, last_id);

                        first_id = id;
                        last_id = id;
                    }
                    copy_experts(first_id, last_id);
                } else {
                    // try async copy, but if not possible, we can still use a sync copy without synchronizing the dst backend, since we handle the synchronization here with multiple copies and events
                    // TODO: add public function to facilitate this, since applications do not have direct access to the backend interface
                    if (!split_backend->iface.cpy_tensor_async || !split_backend->iface.cpy_tensor_async(input_backend, split_backend, input, input_cpy)) {
                        ggml_backend_synchronize(input_backend);
                        if (sched->events[split_backend_id][split->copy_id] != NULL) {
                            ggml_backend_event_synchronize(sched->events[split_backend_id][split->copy_id]);
                        } else {
                            ggml_backend_synchronize(split_backend);
                        }
                        ggml_backend_tensor_copy(input, input_cpy);
                        if (sched->staging_double_buffer) {
                            ggml_backend_synchronize(split_backend);
                        }
                    }
                }
            }
            if (weight_cached) {
                weight_cache_entry->valid = true;
                if (!ggml_backend_sched_weight_cache_alias_output_enabled() && sched->weight_cache_prefill) {
                    const bool filled_output = strcmp(input->name, "output.weight") == 0;
                    const bool filled_gate = strstr(input->name, ".ffn_gate.weight") != NULL &&
                                             strncmp(input->name, "blk.", 4) == 0;
                    if (filled_output) {
                        // output.weight overlaps every cached gate at offset 0.
                        for (int i = 0; i < sched->weight_cache_n; ++i) {
                            if (&sched->weight_cache_entries[i] != weight_cache_entry) {
                                sched->weight_cache_entries[i].valid = false;
                            }
                        }
                    } else if (filled_gate) {
                        // A new prompt graph has overwritten a portion of the
                        // previous output matrix, so force output to refill.
                        for (int i = 0; i < sched->weight_cache_n; ++i) {
                            if (strcmp(sched->weight_cache_entries[i].src->name, "output.weight") == 0) {
                                sched->weight_cache_entries[i].valid = false;
                            }
                        }
                    }
                }
                if (input_cpy->view_src == sched->weight_cache_alias_output) {
                    sched->weight_cache_alias_dirty = true;
                }
                if (getenv("LLAMA_WEIGHT_CACHE_TRACE") != NULL) {
                    fprintf(stderr, "WEIGHT_CACHE_FILL: %s offset=%zu bytes=%zu\n",
                            input->name, weight_cache_entry->offset, ggml_nbytes(input));
                    fflush(stderr);
                }
            }
        }

        if (!sched->callback_eval) {
            if (sched->staging_double_buffer && getenv("LLAMA_STAGING_TRACE") != NULL) {
                const ggml_tensor * first = split->graph.n_nodes > 0 ? split->graph.nodes[0] : NULL;
                fprintf(stderr, "STAGING_SPLIT: id=%d copy=%d staging=%d nodes=%d first=%s op=%s inputs=%d\n",
                        split_id, split->copy_id, split->staging_id, split->graph.n_nodes,
                        first ? first->name : "<none>", first ? ggml_op_name(first->op) : "<none>", split->n_inputs);
                for (int ti = 0; ti < split->n_inputs; ++ti) {
                    fprintf(stderr, "  STAGING_INPUT: %s bytes=%zu\n",
                            split->inputs[ti]->name, ggml_nbytes(split->inputs[ti]));
                }
                fflush(stderr);
            }
            bool timed_cpu_gate = false;
            bool decode_cpu_gate = false;
            if (cpu_gate_timing &&
                    ggml_backend_dev_type(ggml_backend_get_device(split_backend)) == GGML_BACKEND_DEVICE_TYPE_CPU) {
                for (int node_id = 0; node_id < split->graph.n_nodes && !timed_cpu_gate; ++node_id) {
                    const struct ggml_tensor * node = split->graph.nodes[node_id];
                    if (ggml_backend_sched_op_batch_size(node) >= 32) {
                        continue;
                    }
                    for (int src_id = 0; src_id < GGML_MAX_SRC; ++src_id) {
                        const struct ggml_tensor * src = node->src[src_id];
                        if (src != NULL && strstr(src->name, ".ffn_gate.weight") != NULL) {
                            timed_cpu_gate = true;
                            break;
                        }
                    }
                }
            }
            int decode_cpu_gate_weight_src = -1;
            if ((cpu_gpu_ffn_overlap || cpu_gate_ram_mirror || cpu_gpu_gate_row_split > 0) &&
                    ggml_backend_dev_type(ggml_backend_get_device(split_backend)) == GGML_BACKEND_DEVICE_TYPE_CPU &&
                    split->graph.n_nodes == 1 &&
                    ggml_backend_sched_op_batch_size(split->graph.nodes[0]) < 32) {
                const struct ggml_tensor * node = split->graph.nodes[0];
                for (int src_id = 0; src_id < GGML_MAX_SRC; ++src_id) {
                    const struct ggml_tensor * src = node->src[src_id];
                    if (src != NULL && strstr(src->name, ".ffn_gate.weight") != NULL) {
                        decode_cpu_gate = true;
                        decode_cpu_gate_weight_src = src_id;
                        break;
                    }
                }
            }
            struct ggml_tensor * decode_gate_weight = decode_cpu_gate_weight_src >= 0 ?
                    split->graph.nodes[0]->src[decode_cpu_gate_weight_src] : nullptr;
            void * decode_gate_weight_original_data = decode_gate_weight != nullptr ? decode_gate_weight->data : nullptr;
            if (decode_cpu_gate && cpu_gate_ram_mirror && cpu_gpu_gate_row_split <= 0) {
                void * mirror_data = ggml_backend_sched_gate_ram_mirror(sched, decode_gate_weight);
                if (mirror_data == nullptr) {
                    return GGML_STATUS_ALLOC_FAILED;
                }
                decode_gate_weight->data = mirror_data;
            }
            const int64_t cpu_gate_begin_us = timed_cpu_gate ? ggml_time_us() : 0;
            enum ggml_status ec = GGML_STATUS_SUCCESS;

            if (decode_cpu_gate && cpu_gpu_gate_row_split > 0) {
                GGML_ASSERT(pending_cpu_gate == nullptr && !pending_gate_split.active);
                ec = ggml_backend_sched_compute_gate_row_split(
                        sched, split, split_backend_id,
                        decode_cpu_gate_weight_src, cpu_gpu_gate_row_split,
                        &pending_gate_split, &gate_prefetch);
                if (ec == GGML_STATUS_SUCCESS) {
                    pending_cpu_gate = split->graph.nodes[split->graph.n_nodes - 1];
                    pending_cpu_backend_id = split_backend_id;
                }
            } else if (decode_cpu_gate && cpu_gpu_ffn_overlap) {
                GGML_ASSERT(pending_cpu_gate == nullptr);
                sched->cpu_gpu_overlap->submit(split_backend, split->graph);
                pending_cpu_gate = split->graph.nodes[split->graph.n_nodes - 1];
                pending_cpu_backend_id = split_backend_id;
                pending_cpu_gate_weight = decode_gate_weight;
                pending_cpu_gate_weight_data = decode_gate_weight_original_data;
            } else if (overlap_gate_input_id >= 0) {
                struct ggml_tensor * gate_input = split->inputs[overlap_gate_input_id];
                struct ggml_tensor * gate_input_cpy = tensor_copy(
                        gate_input, split_backend_id, split->copy_id);
                struct ggml_backend_sched_gate_row_split_plan * completed_gate_plan =
                        pending_gate_split.active ? pending_gate_split.plan : nullptr;
                const int completed_gate_slot = pending_gate_split.active ? pending_gate_split.slot : -1;

                int prefix_end = 0;
                for (; prefix_end < split->graph.n_nodes; ++prefix_end) {
                    const struct ggml_tensor * node = split->graph.nodes[prefix_end];
                    bool consumes_gate = false;
                    for (int src_id = 0; src_id < GGML_MAX_SRC; ++src_id) {
                        if (node->src[src_id] == gate_input_cpy) {
                            consumes_gate = true;
                            break;
                        }
                    }
                    if (consumes_gate) {
                        break;
                    }
                }

                if (prefix_end > 0) {
                    struct ggml_cgraph prefix = ggml_graph_view(&split->graph, 0, prefix_end);
                    ec = ggml_backend_graph_compute_async(split_backend, &prefix);
                }

                if (ec == GGML_STATUS_SUCCESS) {
                    bool direct_gate_gpu_ready = false;
                    ec = wait_pending_cpu_gate(gate_input_cpy, &direct_gate_gpu_ready);

                    if (ec == GGML_STATUS_SUCCESS && !direct_gate_gpu_ready) {
                        ggml_backend_t input_backend = ggml_backend_sched_get_tensor_backend(sched, gate_input);
                        // The custom CUDA staging path enqueues the H2D copy on its
                        // copy stream and inserts a wait after the already queued
                        // FFN-up prefix.  Its synchronous fallback is still safe.
                        if (!split_backend->iface.cpy_tensor_async ||
                                !split_backend->iface.cpy_tensor_async(
                                        input_backend, split_backend, gate_input, gate_input_cpy)) {
                            ggml_backend_synchronize(input_backend);
                            ggml_backend_synchronize(split_backend);
                            ggml_backend_tensor_copy(gate_input, gate_input_cpy);
                        }
                    }
                }

                if (ec == GGML_STATUS_SUCCESS && prefix_end < split->graph.n_nodes) {
                    struct ggml_cgraph suffix = ggml_graph_view(
                            &split->graph, prefix_end, split->graph.n_nodes);
                    ec = ggml_backend_graph_compute_async(split_backend, &suffix);
                }

                // The complete GPU suffix is now queued.  Prefetch the GPU
                // rows of the next host gate into the alternate buffer.  The
                // CUDA copy stream runs beside the current layer and records a
                // wait behind its already queued work, removing PCIe transfer
                // from the next gate's critical path.
                if (ec == GGML_STATUS_SUCCESS && completed_gate_plan != nullptr && completed_gate_slot >= 0) {
                    for (int future_id = split_id + 1; future_id < sched->n_splits; ++future_id) {
                        struct ggml_backend_sched_split * future = &splits[future_id];
                        if (ggml_backend_dev_type(ggml_backend_get_device(sched->backends[future->backend_id])) !=
                                GGML_BACKEND_DEVICE_TYPE_CPU || future->graph.n_nodes != 1 ||
                                ggml_backend_sched_op_batch_size(future->graph.nodes[0]) >= 32) {
                            continue;
                        }
                        struct ggml_tensor * future_weight = nullptr;
                        for (int src_id = 0; src_id < GGML_MAX_SRC; ++src_id) {
                            struct ggml_tensor * src = future->graph.nodes[0]->src[src_id];
                            if (src != nullptr && strstr(src->name, ".ffn_gate.weight") != nullptr) {
                                future_weight = src;
                                break;
                            }
                        }
                        if (future_weight == nullptr) {
                            continue;
                        }
                        if (future_weight->ne[0] != completed_gate_plan->n_in ||
                                future_weight->ne[1] != completed_gate_plan->n_out) {
                            break;
                        }

                        const int prefetch_slot = 1 - completed_gate_slot;
                        struct ggml_tensor future_weight_src = *future_weight;
                        future_weight_src.ne[1] = completed_gate_plan->n_gpu;
                        future_weight_src.ne[2] = 1;
                        future_weight_src.ne[3] = 1;
                        snprintf(future_weight_src.name, sizeof(future_weight_src.name),
                                "%s.row_split_prefetch", future_weight->name);
                        ggml_backend_t future_cpu_backend = sched->backends[future->backend_id];
                        const struct ggml_tensor * future_weight_copy_src = &future_weight_src;
                        if (const struct ggml_tensor * mirror =
                                ggml_backend_sched_gate_wc_mirror(
                                        sched, split_backend, future_weight,
                                        completed_gate_plan->n_gpu)) {
                            future_weight_copy_src = mirror;
                        }
                        const bool copied_async = split_backend->iface.cpy_tensor_async &&
                                split_backend->iface.cpy_tensor_async(
                                        future_cpu_backend, split_backend, future_weight_copy_src,
                                        completed_gate_plan->weight_gpu[prefetch_slot]);
                        if (copied_async) {
                            gate_prefetch.weight = future_weight;
                            gate_prefetch.n_gpu = completed_gate_plan->n_gpu;
                            gate_prefetch.slot = prefetch_slot;
                        }
                        break;
                    }
                }

                ++overlap_pairs;
                overlap_prefix_nodes += prefix_end;
                if (cpu_gpu_ffn_overlap_trace && overlap_pairs <= 4) {
                    fprintf(stderr,
                            "CPU_GPU_FFN_OVERLAP: pair=%d gate=%s prefix_nodes=%d total_nodes=%d status=%d\n",
                            overlap_pairs, gate_input->name, prefix_end, split->graph.n_nodes, (int) ec);
                    fflush(stderr);
                }
            } else {
                ec = ggml_backend_graph_compute_async(split_backend, &split->graph);
                if (decode_gate_weight != nullptr && cpu_gate_ram_mirror) {
                    decode_gate_weight->data = decode_gate_weight_original_data;
                }
            }
            if (timed_cpu_gate) {
                cpu_gate_total_us += ggml_time_us() - cpu_gate_begin_us;
                ++cpu_gate_splits;
            }
            if (ec != GGML_STATUS_SUCCESS) {
                return ec;
            }
        } else {
            // similar to ggml_backend_compare_graph_backend
            for (int j0 = 0; j0 < split->graph.n_nodes; j0++) {
                struct ggml_tensor * t = split->graph.nodes[j0];

                // check if the user needs data from this node
                bool need = sched->callback_eval(t, true, sched->callback_eval_user_data);

                int j1 = j0;

                // determine the range [j0, j1] of nodes that can be computed together
                while (!need && j1 < split->graph.n_nodes - 1) {
                    t = split->graph.nodes[++j1];
                    need = sched->callback_eval(t, true, sched->callback_eval_user_data);
                }

                struct ggml_cgraph gv = ggml_graph_view(&split->graph, j0, j1 + 1);

                enum ggml_status ec = ggml_backend_graph_compute_async(split_backend, &gv);
                if (ec != GGML_STATUS_SUCCESS) {
                    return ec;
                }

                // TODO: pass backend to the callback, then the user can decide if they want to synchronize
                ggml_backend_synchronize(split_backend);

                if (need && !sched->callback_eval(t, false, sched->callback_eval_user_data)) {
                    break;
                }

                j0 = j1;
            }
        }

        // record the event of this split
        const bool split_is_pending_cpu_gate =
                pending_cpu_gate != nullptr && pending_cpu_backend_id == split_backend_id &&
                split->graph.n_nodes == 1 && split->graph.nodes[0] == pending_cpu_gate;
        if (!split_is_pending_cpu_gate && sched->events[split_backend_id][split->copy_id] != NULL) {
            ggml_backend_event_record(sched->events[split_backend_id][split->copy_id], split_backend);
        }
        if (!split_is_pending_cpu_gate && sched->staging_double_buffer && split->staging_id >= 0 &&
                sched->staging_events[split_backend_id][split->staging_id] != NULL) {
            ggml_backend_event_record(sched->staging_events[split_backend_id][split->staging_id], split_backend);
            sched->staging_event_recorded[split_backend_id][split->staging_id] = true;
        }

        prev_backend_id = split_backend_id;
        prev_copy_id = split->copy_id;
    }

    const enum ggml_status pending_status = wait_pending_cpu_gate();
    if (pending_status != GGML_STATUS_SUCCESS) {
        return pending_status;
    }

    if (cpu_gpu_ffn_overlap_trace && overlap_pairs > 0) {
        fprintf(stderr, "CPU_GPU_FFN_OVERLAP_SUMMARY: pairs=%d prefix_nodes=%d\n",
                overlap_pairs, overlap_prefix_nodes);
        fflush(stderr);
    }

    if (cpu_gate_timing && cpu_gate_splits > 0) {
        fprintf(stderr, "CPU_GATE_TIMING: splits=%d total_ms=%.3f\n",
                cpu_gate_splits, cpu_gate_total_us/1000.0);
        fflush(stderr);
    }
    return GGML_STATUS_SUCCESS;
}

ggml_backend_sched_t ggml_backend_sched_new(
        ggml_backend_t * backends,
        ggml_backend_buffer_type_t * bufts,
        int n_backends,
        size_t graph_size,
        bool parallel,
        bool op_offload) {
    GGML_ASSERT(n_backends > 0);
    GGML_ASSERT(n_backends <= GGML_SCHED_MAX_BACKENDS);
    GGML_ASSERT(ggml_backend_dev_type(ggml_backend_get_device(backends[n_backends - 1])) == GGML_BACKEND_DEVICE_TYPE_CPU);

    struct ggml_backend_sched * sched = (ggml_backend_sched *) calloc(1, sizeof(struct ggml_backend_sched));

    if (getenv("LLAMA_CPU_GPU_FFN_OVERLAP") != NULL ||
            getenv("LLAMA_CPU_GPU_GATE_ROW_SPLIT") != NULL) {
        sched->cpu_gpu_overlap = new ggml_backend_sched_cpu_gpu_overlap();
    }

    const char * GGML_SCHED_DEBUG = getenv("GGML_SCHED_DEBUG");
    sched->debug = GGML_SCHED_DEBUG ? atoi(GGML_SCHED_DEBUG) : 0;

    sched->debug_realloc = 0;
#ifdef GGML_SCHED_NO_REALLOC
    sched->debug_realloc = 1;
#endif
    const char * GGML_SCHED_DEBUG_REALLOC = getenv("GGML_SCHED_DEBUG_REALLOC");
    sched->debug_realloc = GGML_SCHED_DEBUG_REALLOC ? atoi(GGML_SCHED_DEBUG_REALLOC) : sched->debug_realloc;

    sched->n_backends = n_backends;
    sched->staging_double_buffer = !parallel && ggml_backend_sched_staging_double_buffer_enabled();
    sched->staging_slots = 2;
    if (const char * value = getenv("LLAMA_STAGING_SLOTS")) {
        sched->staging_slots = std::max(2, std::min(GGML_SCHED_MAX_STAGING_SLOTS, atoi(value)));
    }
    // Staging ping-pong is independent from pipeline copies. Reusing copy_id for
    // per-split staging corrupts ordinary cross-split activation copies.
    sched->n_copies = parallel ? GGML_SCHED_MAX_COPIES : 1;
    memset(sched->staging_buffers, 0, sizeof(sched->staging_buffers));
    memset(sched->weight_cache_buffers, 0, sizeof(sched->weight_cache_buffers));
    sched->weight_cache_capacity = 128;
    sched->weight_cache_entries = (ggml_backend_sched_weight_cache_entry *)
            calloc(sched->weight_cache_capacity, sizeof(sched->weight_cache_entries[0]));
    GGML_ASSERT(sched->weight_cache_entries != NULL);
    sched->gate_ram_mirror_capacity = 128;
    sched->gate_ram_mirrors = (ggml_backend_sched_gate_ram_mirror_entry *)
            calloc(sched->gate_ram_mirror_capacity, sizeof(sched->gate_ram_mirrors[0]));
    GGML_ASSERT(sched->gate_ram_mirrors != NULL);
    sched->gate_wc_mirror_capacity = 128;
    sched->gate_wc_mirrors = (ggml_backend_sched_gate_wc_mirror_entry *)
            calloc(sched->gate_wc_mirror_capacity, sizeof(sched->gate_wc_mirrors[0]));
    GGML_ASSERT(sched->gate_wc_mirrors != NULL);

    // initialize hash table
    // FIXME: needs to be size*2 to account for leafs (do it in graph_split instead)
    sched->hash_set    = ggml_hash_set_new(graph_size);
    sched->hv_tensor_backend_ids = (int *) malloc(sched->hash_set.size * sizeof(sched->hv_tensor_backend_ids[0]));
    sched->hv_tensor_copies      = (ggml_tensor **) malloc(sched->hash_set.size * sched->n_backends * sched->n_copies * sizeof(struct ggml_tensor *));

    const size_t ggml_sched_max_splits = graph_size; // at most there is one split for each node in the graph
    const size_t nodes_size = graph_size + ggml_sched_max_splits*GGML_SCHED_MAX_SPLIT_INPUTS*2;
    sched->node_backend_ids = (int *) calloc(nodes_size, sizeof(sched->node_backend_ids[0]));
    sched->leaf_backend_ids = (int *) calloc(nodes_size, sizeof(sched->leaf_backend_ids[0]));
    sched->prev_node_backend_ids = (int *) calloc(nodes_size, sizeof(sched->prev_node_backend_ids[0]));
    sched->prev_leaf_backend_ids = (int *) calloc(nodes_size, sizeof(sched->prev_leaf_backend_ids[0]));

    sched->debug_graph_size = 0;
    sched->debug_prev_graph_size = 0;

    sched->context_buffer_size = ggml_sched_max_splits*GGML_SCHED_MAX_SPLIT_INPUTS*2*sizeof(struct ggml_tensor) + ggml_graph_overhead_custom(graph_size, false);
    sched->context_buffer = (char *) malloc(sched->context_buffer_size);

    const int initial_splits_capacity = 16;
    sched->splits = (ggml_backend_sched_split *) calloc(initial_splits_capacity, sizeof(sched->splits[0]));
    sched->splits_capacity = initial_splits_capacity;

    sched->graph_inputs_capacity = GGML_SCHED_MAX_SPLIT_INPUTS;
    sched->graph_inputs = (struct ggml_tensor **) calloc(sched->graph_inputs_capacity, sizeof(struct ggml_tensor *));

    for (int b = 0; b < n_backends; b++) {
        sched->backends[b] = backends[b];
        sched->bufts[b] = bufts ? bufts[b] : ggml_backend_get_default_buffer_type(backends[b]);
        GGML_ASSERT(ggml_backend_supports_buft(backends[b], sched->bufts[b]));

        if (sched->n_copies > 1) {
            for (int c = 0; c < sched->n_copies; c++) {
                sched->events[b][c] = ggml_backend_event_new(backends[b]->device);
            }
        }
        if (sched->staging_double_buffer) {
            for (int c = 0; c < sched->staging_slots; c++) {
                sched->staging_events[b][c] = ggml_backend_event_new(backends[b]->device);
                sched->staging_event_recorded[b][c] = false;
            }
        }
    }

    sched->galloc = ggml_gallocr_new_n(sched->bufts, n_backends);
    sched->op_offload = op_offload;

    ggml_backend_sched_reset(sched);

    return sched;
}

void ggml_backend_sched_free(ggml_backend_sched_t sched) {
    if (sched == NULL) {
        return;
    }
    delete sched->cpu_gpu_overlap;
    sched->cpu_gpu_overlap = nullptr;
    for (int i = 0; i < sched->gate_ram_mirror_n; ++i) {
        ggml_aligned_free(sched->gate_ram_mirrors[i].data, sched->gate_ram_mirrors[i].size);
        sched->gate_ram_mirrors[i].data = nullptr;
    }
    for (int i = 0; i < sched->gate_wc_mirror_n; ++i) {
        ggml_backend_buffer_free(sched->gate_wc_mirrors[i].buffer);
        sched->gate_wc_mirrors[i].buffer = nullptr;
    }
    ggml_backend_buffer_free(sched->gate_split_gpu_weight_buffer[0]);
    ggml_backend_buffer_free(sched->gate_split_gpu_weight_buffer[1]);
    ggml_backend_buffer_free(sched->gate_split_gpu_output_buffer);
    ggml_backend_buffer_free(sched->gate_split_cpu_output_buffer);
    free(sched->gate_split_gpu_output_host);
    for (int i = 0; i < 32; ++i) {
        if (sched->gate_split_plans[i] != nullptr) {
            ggml_free(sched->gate_split_plans[i]->ctx);
            delete sched->gate_split_plans[i];
            sched->gate_split_plans[i] = nullptr;
        }
    }
    for (int b = 0; b < sched->n_backends; b++) {
        for (int c = 0; c < sched->n_copies; c++) {
            ggml_backend_event_free(sched->events[b][c]);
        }
        for (int c = 0; c < sched->staging_slots; c++) {
            ggml_backend_event_free(sched->staging_events[b][c]);
            if (sched->staging_buffers[b][c] != NULL) {
                ggml_backend_buffer_free(sched->staging_buffers[b][c]);
                sched->staging_buffers[b][c] = NULL;
            }
        }
        if (sched->weight_cache_buffers[b] != NULL) {
            ggml_backend_buffer_free(sched->weight_cache_buffers[b]);
            sched->weight_cache_buffers[b] = NULL;
        }
    }
    ggml_gallocr_free(sched->galloc);
    ggml_free(sched->ctx);
    ggml_hash_set_free(&sched->hash_set);
    for (int i = 0; i < sched->splits_capacity; i++) {
        free(sched->splits[i].inputs);
    }
    free(sched->splits);
    free(sched->graph_inputs);
    free(sched->hv_tensor_backend_ids);
    free(sched->hv_tensor_copies);
    free(sched->node_backend_ids);
    free(sched->leaf_backend_ids);
    free(sched->prev_node_backend_ids);
    free(sched->prev_leaf_backend_ids);
    free(sched->context_buffer);
    free(sched->graph.nodes);
    free(sched->graph.leafs);
    free(sched->weight_cache_entries);
    free(sched->gate_ram_mirrors);
    free(sched->gate_wc_mirrors);
    free(sched->weight_cache_alias_shadow);
    free(sched);
}

bool ggml_backend_sched_share_compute_buffers(ggml_backend_sched_t dst, ggml_backend_sched_t src) {
    if (dst == nullptr || src == nullptr || dst == src ||
        dst->n_copies != src->n_copies ||
        dst->n_backends != src->n_backends || dst->is_alloc) {
        return false;
    }

    for (int i = 0; i < dst->n_backends; ++i) {
        if (dst->bufts[i] != src->bufts[i] ||
            ggml_backend_get_device(dst->backends[i]) != ggml_backend_get_device(src->backends[i])) {
            return false;
        }
    }

    return ggml_gallocr_share_buffers(dst->galloc, src->galloc);
}

void ggml_backend_sched_reset(ggml_backend_sched_t sched) {
    GGML_ASSERT(sched);
    // reset state for the next run
    if (!sched->is_reset) {
        ggml_hash_set_reset(&sched->hash_set);
        memset(sched->hv_tensor_backend_ids, -1, sched->hash_set.size * sizeof(sched->hv_tensor_backend_ids[0]));
        memset(sched->hv_tensor_copies,       0, sched->hash_set.size * sched->n_backends * sched->n_copies * sizeof(struct ggml_tensor *));
        sched->is_reset = true;
    }
    sched->is_alloc = false;
}

void ggml_backend_sched_request_reset(ggml_backend_sched_t sched) {
    GGML_ASSERT(sched);

    // Sparse page-table changes must happen only after every graph and staging
    // stream using the arena has completed.
    ggml_backend_sched_synchronize(sched);
    ggml_gallocr_request_reset(sched->galloc);
    ggml_backend_sched_reset(sched);
}

void ggml_backend_sched_reserve_size(ggml_backend_sched_t sched, struct ggml_cgraph * measure_graph, size_t * sizes) {
    GGML_ASSERT(sched);
    GGML_ASSERT((int)sched->hash_set.size >= measure_graph->n_nodes + measure_graph->n_leafs);
    GGML_ASSERT(sizes);

    ggml_backend_sched_reset(sched);

    ggml_backend_sched_synchronize(sched);

    ggml_backend_sched_split_graph(sched, measure_graph);

    ggml_gallocr_reserve_n_size(sched->galloc, &sched->graph, sched->node_backend_ids, sched->leaf_backend_ids, sizes);
}

bool ggml_backend_sched_reserve(ggml_backend_sched_t sched, struct ggml_cgraph * measure_graph) {
    GGML_ASSERT(sched);
    GGML_ASSERT((int)sched->hash_set.size >= measure_graph->n_nodes + measure_graph->n_leafs);

    ggml_backend_sched_synchronize(sched);

    ggml_backend_sched_split_graph(sched, measure_graph);

    if (!ggml_gallocr_reserve_n(sched->galloc, &sched->graph, sched->node_backend_ids, sched->leaf_backend_ids)) {
        return false;
    }

    ggml_backend_sched_reset(sched);

    return true;
}

bool ggml_backend_sched_alloc_graph(ggml_backend_sched_t sched, struct ggml_cgraph * graph) {
    GGML_ASSERT(sched);
    GGML_ASSERT((int)sched->hash_set.size >= graph->n_nodes + graph->n_leafs);
    GGML_ASSERT(!sched->is_alloc);

    sched->cur_copy = sched->next_copy;
    sched->next_copy = (sched->next_copy + 1) % sched->n_copies;

    ggml_backend_sched_split_graph(sched, graph);

    if (!ggml_backend_sched_alloc_splits(sched)) {
        return false;
    }

    sched->is_alloc = true;

    return true;
}

enum ggml_status ggml_backend_sched_graph_compute(ggml_backend_sched_t sched, struct ggml_cgraph * graph) {
    enum ggml_status err = ggml_backend_sched_graph_compute_async(sched, graph);
    ggml_backend_sched_synchronize(sched);
    return err;
}

enum ggml_status ggml_backend_sched_graph_compute_async(ggml_backend_sched_t sched, struct ggml_cgraph * graph) {
    GGML_ASSERT(sched);
    if (!sched->is_reset && !sched->is_alloc) {
        ggml_backend_sched_reset(sched);
    }

    if (!sched->is_alloc) {
        if (!ggml_backend_sched_alloc_graph(sched, graph)) {
            return GGML_STATUS_ALLOC_FAILED;
        }
    }

    return ggml_backend_sched_compute_splits(sched);
}

void ggml_backend_sched_synchronize(ggml_backend_sched_t sched) {
    GGML_ASSERT(sched);
    for (int i = 0; i < sched->n_backends; i++) {
        ggml_backend_synchronize(sched->backends[i]);
    }
    if (!sched->is_alloc) {
        // if the graph is not already allocated, always use copy 0 after a synchronization
        // this ensures that during generation the same copy is used every time,
        // which avoids changes in the graph that could cause CUDA or other graphs to be disabled
        sched->next_copy = 0;
    }
}

void ggml_backend_sched_set_eval_callback(ggml_backend_sched_t sched, ggml_backend_sched_eval_callback callback, void * user_data) {
    GGML_ASSERT(sched);
    sched->callback_eval = callback;
    sched->callback_eval_user_data = user_data;
}

int ggml_backend_sched_get_n_splits(ggml_backend_sched_t sched) {
    GGML_ASSERT(sched);
    return sched->n_splits;
}

int ggml_backend_sched_get_n_copies(ggml_backend_sched_t sched) {
    GGML_ASSERT(sched);
    return sched->n_copies;
}

int ggml_backend_sched_get_n_backends(ggml_backend_sched_t sched) {
    GGML_ASSERT(sched);
    return sched->n_backends;
}

ggml_backend_t ggml_backend_sched_get_backend(ggml_backend_sched_t sched, int i) {
    GGML_ASSERT(sched);
    GGML_ASSERT(i >= 0 && i < sched->n_backends);
    return sched->backends[i];
}

ggml_backend_buffer_type_t ggml_backend_sched_get_buffer_type(ggml_backend_sched_t sched, ggml_backend_t backend) {
    GGML_ASSERT(sched);
    int backend_index = ggml_backend_sched_backend_id(sched, backend);
    GGML_ASSERT(backend_index >= 0 && backend_index < sched->n_backends);

    return sched->bufts[backend_index];
}

size_t ggml_backend_sched_get_buffer_size(ggml_backend_sched_t sched, ggml_backend_t backend) {
    GGML_ASSERT(sched);
    int backend_index = ggml_backend_sched_backend_id(sched, backend);
    GGML_ASSERT(backend_index >= 0 && backend_index < sched->n_backends);

    return ggml_gallocr_get_buffer_size(sched->galloc, backend_index);
}

void ggml_backend_sched_set_tensor_backend(ggml_backend_sched_t sched, struct ggml_tensor * node, ggml_backend_t backend) {
    GGML_ASSERT(sched);
    int backend_index = ggml_backend_sched_backend_id(sched, backend);
    GGML_ASSERT(backend_index >= 0 && backend_index < sched->n_backends);
    tensor_backend_id(node) = backend_index;
    SET_CAUSE(node, "usr");
    sched->is_reset = false;
}

ggml_backend_t ggml_backend_sched_get_tensor_backend(ggml_backend_sched_t sched, struct ggml_tensor * node) {
    GGML_ASSERT(sched);
    int backend_index = tensor_backend_id(node);
    if (backend_index == -1) {
        return NULL;
    }
    return sched->backends[backend_index];
}

// utils

enum ggml_status ggml_backend_view_init(struct ggml_tensor * tensor) {
    GGML_ASSERT(tensor);
    GGML_ASSERT(tensor->buffer == NULL);
    GGML_ASSERT(tensor->view_src != NULL);
    GGML_ASSERT(tensor->view_src->buffer != NULL);
    GGML_ASSERT(tensor->view_src->data != NULL);

    tensor->buffer = tensor->view_src->buffer;
    tensor->data = (char *)tensor->view_src->data + tensor->view_offs;
    return ggml_backend_buffer_init_tensor(tensor->buffer, tensor);
}

enum ggml_status ggml_backend_tensor_alloc(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor, void * addr) {
    GGML_ASSERT(tensor);
    GGML_ASSERT(tensor->buffer == NULL);
    GGML_ASSERT(tensor->data == NULL);
    GGML_ASSERT(tensor->view_src == NULL);
    GGML_ASSERT(addr >= ggml_backend_buffer_get_base(buffer));
    GGML_ASSERT(ggml_backend_buffer_is_meta(buffer) ||
        (char *) addr + ggml_backend_buffer_get_alloc_size(buffer, tensor) <=
        (char *) ggml_backend_buffer_get_base(buffer) + ggml_backend_buffer_get_size(buffer));

    tensor->buffer = buffer;
    tensor->data = addr;
    return ggml_backend_buffer_init_tensor(buffer, tensor);
}

static struct ggml_tensor * graph_copy_dup_tensor(struct ggml_hash_set hash_set, struct ggml_tensor ** node_copies,
    struct ggml_context * ctx_allocated, struct ggml_context * ctx_unallocated, struct ggml_tensor * src) {

    GGML_ASSERT(src != NULL);
    GGML_ASSERT(src->data && "graph must be allocated");

    size_t id = ggml_hash_insert(&hash_set, src);
    if (id == GGML_HASHSET_ALREADY_EXISTS) {
        return node_copies[ggml_hash_find(&hash_set, src)];
    }

    struct ggml_tensor * dst = ggml_dup_tensor_layout(src->data && !src->view_src ? ctx_allocated : ctx_unallocated, src);
    if (src->view_src != NULL) {
        dst->view_src = graph_copy_dup_tensor(hash_set, node_copies, ctx_allocated, ctx_unallocated, src->view_src);
        dst->view_offs = src->view_offs;
    }
    dst->op = src->op;
    dst->flags = src->flags;
    memcpy(dst->op_params, src->op_params, sizeof(dst->op_params));
    ggml_set_name(dst, src->name);

    // copy src
    for (int i = 0; i < GGML_MAX_SRC; i++) {
        struct ggml_tensor * s = src->src[i];
        if (s == NULL) {
            continue;
        }
        dst->src[i] = graph_copy_dup_tensor(hash_set, node_copies, ctx_allocated, ctx_unallocated, s);
    }

    node_copies[id] = dst;
    return dst;
}

static void graph_copy_init_tensor(struct ggml_hash_set * hash_set, struct ggml_tensor ** node_copies, bool * node_init, struct ggml_tensor * src) {
    size_t id = ggml_hash_find(hash_set, src);
    if (node_init[id]) {
        return;
    }
    node_init[id] = true;

    struct ggml_tensor * dst = node_copies[id];
    if (dst->view_src != NULL) {
        graph_copy_init_tensor(hash_set, node_copies, node_init, src->view_src);
        enum ggml_status status = ggml_backend_view_init(dst);
        GGML_ASSERT(status == GGML_STATUS_SUCCESS);
    }
    else {
        ggml_backend_tensor_copy(src, dst);
    }

    // init src
    for (int i = 0; i < GGML_MAX_SRC; i++) {
        struct ggml_tensor * s = src->src[i];
        if (s == NULL) {
            continue;
        }
        graph_copy_init_tensor(hash_set, node_copies, node_init, s);
    }
}

struct ggml_backend_graph_copy ggml_backend_graph_copy(ggml_backend_t backend, struct ggml_cgraph * graph) {
    GGML_ASSERT(graph);
    struct ggml_hash_set hash_set = ggml_hash_set_new(graph->visited_hash_set.size);
    struct ggml_tensor ** node_copies = (ggml_tensor **) calloc(hash_set.size, sizeof(node_copies[0])); // NOLINT
    bool * node_init = (bool *) calloc(hash_set.size, sizeof(node_init[0]));

    struct ggml_init_params params = {
        /* .mem_size   = */ ggml_tensor_overhead()*hash_set.size + ggml_graph_overhead_custom(graph->size, false),
        /* .mem_buffer = */ NULL,
        /* .no_alloc   = */ true
    };

    struct ggml_context * ctx_allocated = ggml_init(params);
    struct ggml_context * ctx_unallocated = ggml_init(params);

    if (ctx_allocated == NULL || ctx_unallocated == NULL) {
        GGML_LOG_ERROR("%s: failed to allocate context for graph copy\n", __func__);
        ggml_hash_set_free(&hash_set);
        free(node_copies);
        free(node_init);
        ggml_free(ctx_allocated);
        ggml_free(ctx_unallocated);
        return {
            /* .buffer           = */ NULL,
            /* .ctx_allocated    = */ NULL,
            /* .ctx_unallocated  = */ NULL,
            /* .graph            = */ NULL,
        };
    }

    // dup nodes
    for (int i = 0; i < graph->n_nodes; i++) {
        struct ggml_tensor * node = graph->nodes[i];
        graph_copy_dup_tensor(hash_set, node_copies, ctx_allocated, ctx_unallocated, node);
    }

    // allocate nodes
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx_allocated, backend);
    if (buffer == NULL) {
        GGML_LOG_ERROR("%s: failed to allocate buffer for graph copy\n", __func__);
        ggml_hash_set_free(&hash_set);
        free(node_copies);
        free(node_init);
        ggml_free(ctx_allocated);
        ggml_free(ctx_unallocated);
        return {
            /* .buffer           = */ NULL,
            /* .ctx_allocated    = */ NULL,
            /* .ctx_unallocated  = */ NULL,
            /* .graph            = */ NULL,
        };
    }

    //printf("copy buffer size: %zu MB\n", ggml_backend_buffer_get_size(buffer) / 1024 / 1024);

    // copy data and init views
    for (int i = 0; i < graph->n_nodes; i++) {
        struct ggml_tensor * node = graph->nodes[i];
        graph_copy_init_tensor(&hash_set, node_copies, node_init, node);
    }

    // build graph copy
    struct ggml_cgraph * graph_copy = ggml_new_graph_custom(ctx_allocated, graph->size, false);
    for (int i = 0; i < graph->n_nodes; i++) {
        struct ggml_tensor * node = graph->nodes[i];
        struct ggml_tensor * node_copy = node_copies[ggml_hash_find(&hash_set, node)];
        graph_copy->nodes[i] = node_copy;
    }
    graph_copy->n_nodes = graph->n_nodes;

    ggml_hash_set_free(&hash_set);
    free(node_copies);
    free(node_init);

    return {
        /* .buffer           = */ buffer,
        /* .ctx_allocated    = */ ctx_allocated,
        /* .ctx_unallocated  = */ ctx_unallocated,
        /* .graph            = */ graph_copy,
    };
}

void ggml_backend_graph_copy_free(struct ggml_backend_graph_copy copy) {
    ggml_backend_buffer_free(copy.buffer);
    ggml_free(copy.ctx_allocated);
    ggml_free(copy.ctx_unallocated);
}

bool ggml_backend_compare_graph_backend(ggml_backend_t backend1, ggml_backend_t backend2, struct ggml_cgraph * graph, ggml_backend_eval_callback callback, void * user_data, struct ggml_tensor const * const * test_nodes, size_t num_test_nodes) {
    struct ggml_backend_graph_copy copy = ggml_backend_graph_copy(backend2, graph);
    if (copy.buffer == NULL) {
        return false;
    }

    struct ggml_cgraph * g1 = graph;
    struct ggml_cgraph * g2 = copy.graph;

    assert(g1->n_nodes == g2->n_nodes);

    if (num_test_nodes != 0) {
        GGML_ASSERT(test_nodes);
        // Compute the whole graph and only test the output for specific tensors
        ggml_backend_graph_compute(backend1, g1);
        ggml_backend_graph_compute(backend2, g2);

        bool verified = false;
        for (int i = 0; i < g1->n_nodes; i++) {
            for (size_t j = 0; j < num_test_nodes; ++j) {
                if (g1->nodes[i] == test_nodes[j]) {
                    callback(i, g1->nodes[i], g2->nodes[i], user_data);
                    verified = true;
                }
            }
        }
        GGML_ASSERT(verified);
    } else {
        for (int i = 0; i < g1->n_nodes; i++) {
            struct ggml_tensor * t1 = g1->nodes[i];
            struct ggml_tensor * t2 = g2->nodes[i];

            assert(t1->op == t2->op && ggml_are_same_layout(t1, t2));

            struct ggml_cgraph g1v = ggml_graph_view(g1, i, i + 1);
            struct ggml_cgraph g2v = ggml_graph_view(g2, i, i + 1);

            ggml_backend_graph_compute(backend1, &g1v);
            ggml_backend_graph_compute(backend2, &g2v);

            if (ggml_is_view_op(t1->op)) {
                continue;
            }

            // compare results, calculate rms etc
            if (!callback(i, t1, t2, user_data)) {
                break;
            }
        }
    }
    ggml_backend_graph_copy_free(copy);

    return true;
}

// CPU backend - buffer

static void * ggml_backend_cpu_buffer_get_base(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(buffer);
    uintptr_t data = (uintptr_t)buffer->context;

    // align the buffer
    if (data % TENSOR_ALIGNMENT != 0) {
        data = GGML_PAD(data, TENSOR_ALIGNMENT);
    }

    return (void *)data;
}

static void ggml_backend_cpu_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(buffer);
    ggml_aligned_free(buffer->context, buffer->size);
}

static void ggml_backend_cpu_buffer_memset_tensor(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor, uint8_t value, size_t offset, size_t size) {
    GGML_ASSERT(tensor);
    memset((char *)tensor->data + offset, value, size);

    GGML_UNUSED(buffer);
}

static void ggml_backend_cpu_buffer_set_tensor(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    GGML_ASSERT(tensor);
    memcpy((char *)tensor->data + offset, data, size);

    GGML_UNUSED(buffer);
}

static void ggml_backend_cpu_buffer_get_tensor(ggml_backend_buffer_t buffer, const struct ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    GGML_ASSERT(tensor);
    memcpy(data, (const char *)tensor->data + offset, size);

    GGML_UNUSED(buffer);
}

static bool ggml_backend_cpu_buffer_cpy_tensor(ggml_backend_buffer_t buffer, const struct ggml_tensor * src, struct ggml_tensor * dst) {
    GGML_ASSERT(src);
    if (ggml_backend_buffer_is_host(src->buffer)) {
        memcpy(dst->data, src->data, ggml_nbytes(src));
        return true;
    }
    return false;

    GGML_UNUSED(buffer);
}

static void ggml_backend_cpu_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    GGML_ASSERT(buffer);
    memset(buffer->context, value, buffer->size);
}

static const struct ggml_backend_buffer_i ggml_backend_cpu_buffer_i = {
    /* .free_buffer     = */ ggml_backend_cpu_buffer_free_buffer,
    /* .get_base        = */ ggml_backend_cpu_buffer_get_base,
    /* .init_tensor     = */ NULL, // no initialization required
    /* .memset_tensor   = */ ggml_backend_cpu_buffer_memset_tensor,
    /* .set_tensor      = */ ggml_backend_cpu_buffer_set_tensor,
    /* .get_tensor      = */ ggml_backend_cpu_buffer_get_tensor,
    /* .set_tensor_2d   = */ NULL,
    /* .get_tensor_2d   = */ NULL,
    /* .cpy_tensor      = */ ggml_backend_cpu_buffer_cpy_tensor,
    /* .clear           = */ ggml_backend_cpu_buffer_clear,
    /* .reset           = */ NULL,
};

static const struct ggml_backend_buffer_i ggml_backend_cpu_buffer_from_ptr_i = {
    /* .free_buffer     = */ NULL, // ptr is not owned by the buffer, so it does not need to be freed
    /* .get_base        = */ ggml_backend_cpu_buffer_get_base,
    /* .init_tensor     = */ NULL, // no initialization required
    /* .memset_tensor   = */ ggml_backend_cpu_buffer_memset_tensor,
    /* .set_tensor      = */ ggml_backend_cpu_buffer_set_tensor,
    /* .get_tensor      = */ ggml_backend_cpu_buffer_get_tensor,
    /* .set_tensor_2d   = */ NULL,
    /* .get_tensor_2d   = */ NULL,
    /* .cpy_tensor      = */ ggml_backend_cpu_buffer_cpy_tensor,
    /* .clear           = */ ggml_backend_cpu_buffer_clear,
    /* .reset           = */ NULL,
};

// CPU backend buffer type

// this buffer type is defined here to make it available to all backends

static const char * ggml_backend_cpu_buffer_type_get_name(ggml_backend_buffer_type_t buft) {
    return "CPU";

    GGML_UNUSED(buft);
}

static ggml_backend_buffer_t ggml_backend_cpu_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    void * data = ggml_aligned_malloc(size);

    if (data == NULL) {
        GGML_LOG_ERROR("%s: failed to allocate buffer of size %zu\n", __func__, size);
        return NULL;
    }

    return ggml_backend_buffer_init(buft, ggml_backend_cpu_buffer_i, data, size);
}

static size_t ggml_backend_cpu_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    return TENSOR_ALIGNMENT;

    GGML_UNUSED(buft);
}

static bool ggml_backend_cpu_buffer_type_is_host(ggml_backend_buffer_type_t buft) {
    return true;

    GGML_UNUSED(buft);
}

ggml_backend_buffer_type_t ggml_backend_cpu_buffer_type(void) {
    static struct ggml_backend_buffer_type ggml_backend_cpu_buffer_type = {
        /* .iface   = */ {
            /* .get_name         = */ ggml_backend_cpu_buffer_type_get_name,
            /* .alloc_buffer     = */ ggml_backend_cpu_buffer_type_alloc_buffer,
            /* .get_alignment    = */ ggml_backend_cpu_buffer_type_get_alignment,
            /* .get_max_size     = */ NULL, // defaults to SIZE_MAX
            /* .get_alloc_size   = */ NULL, // defaults to ggml_nbytes
            /* .is_host          = */ ggml_backend_cpu_buffer_type_is_host,
        },
        /* .device  = */ NULL, // FIXME ggml_backend_reg_dev_get(ggml_backend_cpu_reg(), 0),
        /* .context = */ NULL,
    };

    return &ggml_backend_cpu_buffer_type;
}

static const char * ggml_backend_cpu_buffer_from_ptr_type_get_name(ggml_backend_buffer_type_t buft) {
    return "CPU_Mapped";

    GGML_UNUSED(buft);
}

static ggml_backend_buffer_type_t ggml_backend_cpu_buffer_from_ptr_type(void) {
    static struct ggml_backend_buffer_type ggml_backend_cpu_buffer_type = {
        /* .iface   = */ {
            /* .get_name         = */ ggml_backend_cpu_buffer_from_ptr_type_get_name,
            /* .alloc_buffer     = */ ggml_backend_cpu_buffer_type_alloc_buffer,
            /* .get_alignment    = */ ggml_backend_cpu_buffer_type_get_alignment,
            /* .get_max_size     = */ NULL, // defaults to SIZE_MAX
            /* .get_alloc_size   = */ NULL, // defaults to ggml_nbytes
            /* .is_host          = */ ggml_backend_cpu_buffer_type_is_host,
        },
        /* .device  = */ NULL, // FIXME ggml_backend_reg_dev_get(ggml_backend_cpu_reg(), 0),
        /* .context = */ NULL,
    };

    return &ggml_backend_cpu_buffer_type;
}

ggml_backend_buffer_t ggml_backend_cpu_buffer_from_ptr(void * ptr, size_t size) {
    GGML_ASSERT((uintptr_t)ptr % TENSOR_ALIGNMENT == 0 && "buffer pointer must be aligned");
    return ggml_backend_buffer_init(ggml_backend_cpu_buffer_from_ptr_type(), ggml_backend_cpu_buffer_from_ptr_i, ptr, size);
}

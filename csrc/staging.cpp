// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

#include <ATen/ATen.h>
#include <ATen/core/grad_mode.h>
#include <torch/csrc/utils/pybind.h>
#include <pybind11/stl.h>

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <system_error>
#include <tuple>
#include <unordered_map>
#include <vector>

#define PICOJSON_USE_INT64
#define PICOJSON_USE_LOCALE 0
#include "third_party/picojson.h"

namespace py = pybind11;
namespace {
// filename, tensor key, target dtype, pinned memory
using Request = std::tuple<std::string, std::string, at::ScalarType, bool>;

class File {
 public:
  explicit File(const std::string& path) : fd_(::open(path.c_str(), O_RDONLY | O_CLOEXEC)) {
    if (fd_ < 0) throw std::system_error(errno, std::generic_category(), path);
  }
  ~File() { ::close(fd_); }
  File(const File&) = delete;
  File& operator=(const File&) = delete;
  int fd() const { return fd_; }
 private:
  int fd_;
};

void read_exact(int fd, void* destination, uint64_t bytes, uint64_t offset) {
  auto* dst = static_cast<char*>(destination);
  while (bytes) {
    // Linux caps a single read below 2 GiB. Also handle short reads and EINTR.
    const auto count = static_cast<size_t>(std::min<uint64_t>(bytes, 1ULL << 30));
    const auto n = ::pread(fd, dst, count, static_cast<off_t>(offset));
    if (n < 0) {
      if (errno == EINTR) continue;
      throw std::system_error(errno, std::generic_category(), "pread");
    }
    if (!n) throw std::runtime_error("Truncated safetensors payload");
    dst += n;
    offset += n;
    bytes -= n;
  }
}

uint64_t nonnegative_integer(const picojson::value& value) {
  if (!value.is<int64_t>() || value.get<int64_t>() < 0)
    throw std::runtime_error("Expected a nonnegative integer in safetensors header");
  return value.get<int64_t>();
}

std::optional<at::ScalarType> dtype(const std::string& name) {
  static const std::unordered_map<std::string, at::ScalarType> types = {
      {"BOOL", at::kBool}, {"U8", at::kByte}, {"I8", at::kChar},
      {"I16", at::kShort}, {"I32", at::kInt}, {"I64", at::kLong},
      {"U16", at::kUInt16}, {"U32", at::kUInt32}, {"U64", at::kUInt64},
      {"F16", at::kHalf}, {"BF16", at::kBFloat16}, {"F32", at::kFloat},
      {"F64", at::kDouble}, {"C64", at::kComplexFloat},
      {"F8_E4M3", at::kFloat8_e4m3fn}, {"F8_E5M2", at::kFloat8_e5m2},
      {"F8_E4M3FNUZ", at::kFloat8_e4m3fnuz},
      {"F8_E5M2FNUZ", at::kFloat8_e5m2fnuz}};
  const auto it = types.find(name);
  if (it == types.end()) return std::nullopt;
  return it->second;
}

struct TensorInfo {
  std::vector<int64_t> shape;
  uint64_t begin, end;
  std::optional<at::ScalarType> type;
};

// Safetensors forbids duplicate keys. Keep the vendored parser unmodified and
// supply its context interface so duplicate names cannot silently replace data.
class HeaderContext : public picojson::default_parse_context {
 public:
  explicit HeaderContext(picojson::value* out) : default_parse_context(out) {}
  template <typename Iter>
  bool parse_object_item(picojson::input<Iter>& in, const std::string& key) {
    auto& object = out_->get<picojson::object>();
    if (object.count(key)) return false;
    HeaderContext child(&object[key]);
    return picojson::_parse(child, in);
  }
  template <typename Iter>
  bool parse_array_item(picojson::input<Iter>& in, size_t) {
    auto& array = out_->get<picojson::array>();
    array.emplace_back();
    HeaderContext child(&array.back());
    return picojson::_parse(child, in);
  }
};

// Safetensors JSON is shallow. Bound nesting before invoking a recursive parser.
void check_nesting(const std::string& header) {
  int depth = 0;
  bool quoted = false, escaped = false;
  for (char c : header) {
    if (quoted) {
      if (escaped) escaped = false;
      else if (c == '\\') escaped = true;
      else if (c == '"') quoted = false;
    } else if (c == '"') quoted = true;
    else if (c == '{' || c == '[') {
      if (++depth > 16) throw std::runtime_error("Safetensors header nesting too deep");
    } else if (c == '}' || c == ']') --depth;
  }
}

std::unordered_map<std::string, TensorInfo> read_header(int fd, uint64_t& data_start) {
  struct stat st;
  if (::fstat(fd, &st)) throw std::system_error(errno, std::generic_category(), "fstat");
  if (!S_ISREG(st.st_mode) || st.st_size < 8)
    throw std::runtime_error("Expected a regular safetensors file with an 8-byte prefix");
  const auto file_size = static_cast<uint64_t>(st.st_size);
  unsigned char prefix[8];
  read_exact(fd, prefix, 8, 0);
  uint64_t size = 0;
  for (unsigned i = 0; i < 8; ++i) size |= uint64_t(prefix[i]) << (8 * i);
  if (!size || size > 100000000 || size > file_size - 8)
    throw std::runtime_error("Invalid safetensors header length");
  data_start = 8 + size;
  std::string header(size, '\0');
  read_exact(fd, header.data(), size, 8);
  if (header[0] != '{') throw std::runtime_error("Invalid safetensors header");
  check_nesting(header);
  picojson::value root;
  HeaderContext context(&root);
  std::string error;
  auto end = picojson::_parse(context, header.begin(), header.end(), &error);
  if (!error.empty() || !root.is<picojson::object>() ||
      std::any_of(end, header.end(), [](char c) { return c != ' '; }))
    throw std::runtime_error("Invalid safetensors JSON header: " + error);

  std::unordered_map<std::string, TensorInfo> tensors;
  std::vector<std::pair<uint64_t, uint64_t>> ranges;
  for (const auto& [name, entry] : root.get<picojson::object>()) {
    if (name == "__metadata__") {
      if (!entry.is<picojson::object>()) throw std::runtime_error("Invalid metadata");
      for (const auto& item : entry.get<picojson::object>())
        if (!item.second.is<std::string>()) throw std::runtime_error("Invalid metadata");
      continue;
    }
    if (!entry.is<picojson::object>() || !entry.get("dtype").is<std::string>() ||
        !entry.get("shape").is<picojson::array>() ||
        !entry.get("data_offsets").is<picojson::array>())
      throw std::runtime_error("Invalid tensor metadata for " + name);
    const auto& offsets = entry.get("data_offsets").get<picojson::array>();
    if (offsets.size() != 2) throw std::runtime_error("Invalid tensor offsets");
    TensorInfo info;
    info.begin = nonnegative_integer(offsets[0]);
    info.end = nonnegative_integer(offsets[1]);
    if (info.end < info.begin || info.end > file_size - data_start)
      throw std::runtime_error("Tensor offsets outside safetensors payload");
    info.type = dtype(entry.get("dtype").get<std::string>());
    for (const auto& dim : entry.get("shape").get<picojson::array>())
      info.shape.push_back(nonnegative_integer(dim));
    uint64_t numel = 1;
    for (auto dim : info.shape) {
      if (dim && numel > uint64_t(std::numeric_limits<int64_t>::max()) / dim)
        throw std::runtime_error("Tensor shape overflows int64");
      numel *= dim;
    }
    if (info.type) {
      auto width = c10::elementSize(*info.type);
      if (numel > uint64_t(std::numeric_limits<int64_t>::max()) / width ||
          numel * width != info.end - info.begin)
        throw std::runtime_error("Tensor shape/dtype does not match payload size");
    }
    ranges.emplace_back(info.begin, info.end);
    tensors.emplace(name, std::move(info));
  }
  std::sort(ranges.begin(), ranges.end());
  uint64_t cursor = 0;
  for (auto [begin, end] : ranges) {
    if (begin != cursor) throw std::runtime_error("Overlapping or incomplete tensor data");
    cursor = end;
  }
  if (cursor != file_size - data_start)
    throw std::runtime_error("Safetensors payload has trailing data");
  return tensors;
}

std::vector<std::optional<at::Tensor>> read_batch(const std::vector<Request>& requests) {
  // Argument conversion has finished before the call_guard releases the GIL.
  // Everything below uses only C++ objects and native ATen operations.
  at::NoGradGuard no_grad;
  std::vector<std::optional<at::Tensor>> results(requests.size());
  std::unordered_map<std::string, std::vector<size_t>> grouped;
  std::vector<std::string> paths;
  for (size_t i = 0; i < requests.size(); ++i) {
    const auto& path = std::get<0>(requests[i]);
    if (path.find('\0') != std::string::npos) throw std::runtime_error("NUL in filename");
    if (!grouped.count(path)) paths.push_back(path);
    grouped[path].push_back(i);
  }
  // One file descriptor/header at a time, even with thousands of offload files.
  for (const auto& path : paths) {
    try {
      File file(path);
      uint64_t data_start;
      const auto header = read_header(file.fd(), data_start);
      for (auto index : grouped.at(path)) {
        const auto& [filename, name, target_dtype, pinned] = requests[index];
        const auto found = header.find(name);
        if (found == header.end()) throw std::runtime_error("Tensor key not found: " + name);
        const auto& info = found->second;
        if (!info.type) continue;  // Upstream Python reader handles newer dtypes.
        auto options = at::TensorOptions().device(at::kCPU).dtype(target_dtype).pinned_memory(pinned);
        auto output = at::empty(info.shape, options);
        if (*info.type == target_dtype) {
          read_exact(file.fd(), output.data_ptr(), info.end - info.begin, data_start + info.begin);
        } else {
          auto source = at::empty(info.shape, options.dtype(*info.type).pinned_memory(false));
          read_exact(file.fd(), source.data_ptr(), info.end - info.begin, data_start + info.begin);
          output.copy_(source);
        }
        results[index] = std::move(output);
      }
    } catch (const std::exception& e) {
      throw std::runtime_error("Native staging failed for " + path + ": " + e.what());
    }
  }
  return results;
}
}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("read_batch", &read_batch, py::call_guard<py::gil_scoped_release>(),
             "Read a batch of safetensors into fresh CPU tensors with the GIL released");
}

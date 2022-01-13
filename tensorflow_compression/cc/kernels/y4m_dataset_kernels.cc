/* Copyright 2021 Google LLC. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "absl/strings/str_join.h"
#include "tensorflow/core/framework/dataset.h"
#include "tensorflow/core/framework/partial_tensor_shape.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/logging.h"

namespace tensorflow_compression {
namespace {
namespace errors = tensorflow::errors;
using absl::string_view;
using std::string;
using std::vector;
using tensorflow::DataTypeVector;
using tensorflow::DT_UINT8;
using tensorflow::int64;
using tensorflow::mutex;
using tensorflow::mutex_lock;
using tensorflow::Node;
using tensorflow::OpKernelContext;
using tensorflow::PartialTensorShape;
using tensorflow::RandomAccessFile;
using tensorflow::Status;
using tensorflow::Tensor;
using tensorflow::uint64;
using tensorflow::uint8;
using tensorflow::data::DatasetBase;
using tensorflow::data::DatasetIterator;
using tensorflow::data::DatasetOpKernel;
using tensorflow::data::IteratorContext;
using tensorflow::data::SerializationContext;

class Y4MDatasetOp : public DatasetOpKernel {
 public:
  explicit Y4MDatasetOp(tensorflow::OpKernelConstruction* ctx)
      : DatasetOpKernel(ctx) {}

 protected:
  void MakeDataset(OpKernelContext* ctx, DatasetBase** output) override {
    const Tensor* filenames_tensor;
    OP_REQUIRES_OK(ctx, ctx->input("filenames", &filenames_tensor));
    OP_REQUIRES(
        ctx, filenames_tensor->dims() <= 1,
        errors::InvalidArgument("`filenames` must be a scalar or a vector."));

    vector<string> filenames;
    filenames.reserve(filenames_tensor->NumElements());
    for (int i = 0; i < filenames_tensor->NumElements(); ++i) {
      filenames.emplace_back(filenames_tensor->flat<tensorflow::tstring>()(i));
    }

    *output = new Dataset(ctx, std::move(filenames));
  }

 private:
  class Dataset : public DatasetBase {
   public:
    explicit Dataset(OpKernelContext* ctx, vector<string> filenames)
        : DatasetBase(tensorflow::data::DatasetContext(ctx)),
          filenames_(std::move(filenames)) {}

    std::unique_ptr<::tensorflow::data::IteratorBase> MakeIteratorInternal(
        const std::string& prefix) const override {
      return std::unique_ptr<::tensorflow::data::IteratorBase>(
          new Iterator({this, absl::StrCat(prefix, "::Y4M")}));
    }

    const DataTypeVector& output_dtypes() const override {
      static DataTypeVector* dtypes = new DataTypeVector({DT_UINT8, DT_UINT8});
      return *dtypes;
    }

    const vector<PartialTensorShape>& output_shapes() const override {
      static vector<PartialTensorShape>* shapes =
          new vector<PartialTensorShape>{{-1, -1, 1}, {-1, -1, 2}};
      return *shapes;
    }

    string DebugString() const override { return "Y4MDatasetOp::Dataset"; }

    Status InputDatasets(vector<const DatasetBase*>* inputs) const override {
      return Status::OK();
    }

    Status CheckExternalState() const override { return Status::OK(); }

   protected:
    Status AsGraphDefInternal(SerializationContext* ctx,
                              DatasetGraphDefBuilder* b,
                              Node** output) const override {
      Node* filenames = nullptr;
      TF_RETURN_IF_ERROR(b->AddVector(filenames_, &filenames));
      TF_RETURN_IF_ERROR(b->AddDataset(this, {filenames}, output));
      return Status::OK();
    }

   private:
    class Iterator : public DatasetIterator<Dataset> {
     public:
      explicit Iterator(const Params& params)
          : DatasetIterator<Dataset>(params) {}

      Status GetNextInternal(IteratorContext* ctx,
                             vector<Tensor>* out_tensors,
                             bool* end_of_sequence) override {
        mutex_lock l(mu_);

        do {
          if (file_) {
            const string_view frame_header = "FRAME\n";
            size_t frame_size = width_ * height_ * 3;
            int64 cbcr_width = width_;
            int64 cbcr_height = height_;
            string_view frame_buffer;

            if (chroma_format_ == ChromaFormat::I420) {
              frame_size /= 2;
              cbcr_width /= 2;
              cbcr_height /= 2;
            }
            const size_t cbcr_size = cbcr_width * cbcr_height;

            // This is a no-op for the second and subsequent frames.
            buffer_.resize(frame_header.size() + frame_size);

            // Try to read the next frame.
            Status status = file_->Read(file_pos_, buffer_.size(),
                                        &frame_buffer, &buffer_[0]);

            // Yield frame on successful read of a complete frame.
            if (status.ok()) {
              DCHECK_EQ(frame_buffer.size(), buffer_.size());

              if (!absl::ConsumePrefix(&frame_buffer, frame_header)) {
                return errors::InvalidArgument(
                    "Input file '", dataset()->filenames_[file_index_],
                    "' has a FRAME marker at byte ", file_pos_, " which is "
                    "either invalid or has unsupported frame parameters.");
              }

              Tensor y_tensor(ctx->allocator({}), DT_UINT8,
                              {height_, width_, 1});
              Tensor cbcr_tensor(ctx->allocator({}), DT_UINT8,
                                 {cbcr_height, cbcr_width, 2});
              auto flat_y = y_tensor.flat<uint8>();
              auto flat_cbcr = cbcr_tensor.flat<uint8>();
              std::memcpy(flat_y.data(), frame_buffer.data(), flat_y.size());
              frame_buffer.remove_prefix(flat_y.size());
              for (int i = 0; i < cbcr_size; i++) {
                flat_cbcr.data()[2*i] = frame_buffer[i];
                flat_cbcr.data()[2*i+1] = frame_buffer[cbcr_size+i];
              }
              out_tensors->push_back(std::move(y_tensor));
              out_tensors->push_back(std::move(cbcr_tensor));

              file_pos_ += buffer_.size();
              *end_of_sequence = false;
              return status;
            }

            // Catch any other errors than out of range, which needs special
            // treatment.
            if (!errors::IsOutOfRange(status)) {
              return status;
            }

            // If frame buffer is not empty, we just read an incomplete frame
            // (or one that has frame parameters that change its size).
            if (!frame_buffer.empty()) {
              return errors::InvalidArgument(
                  "Input file '", dataset()->filenames_[file_index_],
                  "' has an incomplete or unsupported frame at byte ",
                  file_pos_, ". Expected to read ", buffer_.size(),
                  " bytes, only ", frame_buffer.size(), " were available.");
            }

            // Out of range error with empty frame buffer means correct end of
            // file. Clean up and check for next file.
            file_.reset();
            ++file_index_;
          }

          // Exit if there are no more files to process.
          if (file_index_ >= dataset()->filenames_.size()) {
            *end_of_sequence = true;
            return Status::OK();
          }

          // Open next file.
          TF_RETURN_IF_ERROR(ctx->env()->NewRandomAccessFile(
              dataset()->filenames_[file_index_], &file_));

          // Read and parse header.
          TF_RETURN_IF_ERROR(ReadHeader(*file_, file_index_, buffer_));
          TF_RETURN_IF_ERROR(ParseHeader(buffer_, file_index_, width_, height_,
                                         chroma_format_));
          file_pos_ = buffer_.size();
        } while (true);
      }

     protected:
      Status SaveInternal(
          SerializationContext* ctx,
          tensorflow::data::IteratorStateWriter* writer) override {
        mutex_lock l(mu_);

        TF_RETURN_IF_ERROR(
            writer->WriteScalar(full_name("file_index"), file_index_));
        // We use file_pos == -1 to indicate no file is currently being read.
        const int64 file_pos = file_ ? file_pos_ : -1;
        TF_RETURN_IF_ERROR(
            writer->WriteScalar(full_name("file_pos"), file_pos));

        return Status::OK();
      }

      Status RestoreInternal(
          IteratorContext* ctx,
          tensorflow::data::IteratorStateReader* reader) override {
        mutex_lock l(mu_);
        int64 file_index;
        int64 file_pos;

        TF_RETURN_IF_ERROR(
            reader->ReadScalar(full_name("file_index"), &file_index));
        TF_RETURN_IF_ERROR(
            reader->ReadScalar(full_name("file_pos"), &file_pos));

        file_.reset();
        file_index_ = file_index;
        if (file_pos >= 0) {
          TF_RETURN_IF_ERROR(ctx->env()->NewRandomAccessFile(
              dataset()->filenames_[file_index_], &file_));
          TF_RETURN_IF_ERROR(ReadHeader(*file_, file_index_, buffer_));
          TF_RETURN_IF_ERROR(ParseHeader(buffer_, file_index_, width_, height_,
                                         chroma_format_));
          file_pos_ = file_pos;
        }
        return Status::OK();
      }

     private:
      enum class ChromaFormat { undefined, I420, I444 };

      Status ReadHeader(const RandomAccessFile& file, const size_t file_index,
                        string& header) {
        // 256 bytes should be more than enough in most cases. If not, keep
        // reading chunks until header is complete.
        const size_t chunk_size = 256;
        header.clear();
        do {
          const uint64 offset = header.size();
          header.resize(offset + chunk_size);
          string_view chunk;
          Status status = file.Read(
              offset, chunk_size, &chunk, &header[offset]);
          // End of file error is fine, as long as the header is complete.
          if (!(status.ok() || errors::IsOutOfRange(status))) {
            return status;
          }
          const size_t pos = chunk.find('\n');
          if (pos != chunk.npos) {
            if (&header[offset] != chunk.data()) {
              std::memcpy(&header[offset], chunk.data(), pos + 1);
            }
            header.resize(offset + pos + 1);
            return Status::OK();
          }
          // We reached the end of the file, and the header is not complete.
          if (!status.ok()) {
            return errors::InvalidArgument(
                "Input file '", dataset()->filenames_[file_index],
                "' does not contain a complete Y4M header.");
          }
          if (&header[offset] != chunk.data()) {
            std::memcpy(&header[offset], chunk.data(), chunk_size);
          }
        } while (true);
      }

      Status ParseHeader(string_view header, const size_t file_index,
                         int64& width, int64& height,
                         ChromaFormat& chroma_format) {
        const string_view digits = "0123456789";

        width = 0;
        height = 0;
        chroma_format = ChromaFormat::undefined;

        // Last character is guaranteed to be newline because ReadHeader uses
        // it to find the end of the header.
        header.remove_suffix(1);

        if (!absl::ConsumePrefix(&header, "YUV4MPEG2")) {
          return errors::InvalidArgument(
              "Input file '", dataset()->filenames_[file_index],
              "' does not have a YUV4MPEG2 marker.");
        }

        while (!header.empty()) {
          size_t pos;
          if (header.size() < 2 || header[0] != ' ') {
            return errors::InvalidArgument(
                "Input file '", dataset()->filenames_[file_index],
                "' has an invalid Y4M header. Remaining header: '", header,
                "'.");
          }
          const char key = header[1];
          header.remove_prefix(2);
          switch (key) {
            case 'W':
              pos = header.find_first_not_of(digits);
              if (!absl::SimpleAtoi(header.substr(0, pos), &width) ||
                  width <= 0) {
                return errors::InvalidArgument(
                    "Input file '", dataset()->filenames_[file_index],
                    "' has an invalid width specifier '", header.substr(0, pos),
                    "'.");
              }
              header.remove_prefix(pos == header.npos ? header.size() : pos);
              break;
            case 'H':
              pos = header.find_first_not_of(digits);
              if (!absl::SimpleAtoi(header.substr(0, pos), &height) ||
                  height <= 0) {
                return errors::InvalidArgument(
                    "Input file '", dataset()->filenames_[file_index],
                    "' has an invalid height specifier '",
                    header.substr(0, pos), "'.");
              }
              header.remove_prefix(pos == header.npos ? header.size() : pos);
              break;
            case 'C':
              if (absl::ConsumePrefix(&header, "420jpeg")) {
                chroma_format = ChromaFormat::I420;
              } else if (absl::ConsumePrefix(&header, "444")) {
                chroma_format = ChromaFormat::I444;
              } else {
                return errors::InvalidArgument(
                    "Input file '", dataset()->filenames_[file_index],
                    "' has an unsupported chroma format '",
                    header.substr(0, header.find(' ')), "'.");
              }
              break;
            case 'I':
              if (!absl::ConsumePrefix(&header, "p")) {
                return errors::InvalidArgument(
                    "Input file '", dataset()->filenames_[file_index],
                    "' is not in progressive format.");
              }
              break;
            default:
              pos = header.find(' ');
              header.remove_prefix(pos == header.npos ? header.size() : pos);
              break;
          }
        }

        if (!width) {
          return errors::InvalidArgument(
              "Input file '", dataset()->filenames_[file_index],
              "' has no width specifier.");
        }
        if (!height) {
          return errors::InvalidArgument(
              "Input file '", dataset()->filenames_[file_index],
              "' has no height specifier.");
        }
        if (chroma_format == ChromaFormat::undefined) {
          return errors::InvalidArgument(
              "Input file '", dataset()->filenames_[file_index],
              "' has no chroma format specifier.");
        }
        if (chroma_format == ChromaFormat::I420 && (width & 1 || height & 1)) {
          return errors::InvalidArgument(
              "Input file '", dataset()->filenames_[file_index],
              "' has 4:2:0 chroma format, but odd width or height.");
        }
        return Status::OK();
      }

      mutex mu_;
      size_t file_index_ TF_GUARDED_BY(mu_) = 0;
      std::unique_ptr<RandomAccessFile> file_ TF_GUARDED_BY(mu_);
      uint64 file_pos_ TF_GUARDED_BY(mu_);
      string buffer_ TF_GUARDED_BY(mu_);
      int64 width_ TF_GUARDED_BY(mu_);
      int64 height_ TF_GUARDED_BY(mu_);
      ChromaFormat chroma_format_ TF_GUARDED_BY(mu_);
    };

    const vector<string> filenames_;
  };
};

REGISTER_KERNEL_BUILDER(Name("Y4MDataset").Device(tensorflow::DEVICE_CPU),
                        Y4MDatasetOp);

}  // namespace
}  // namespace tensorflow_compression

# Lab 21 — Evaluation Report

**Học viên**: Nguyễn Xuân Hải - 2A202600245  
**Ngày nộp**: 07-05-2026  
**Submission option**: B

## 1. Setup
- **Base model**: `unsloth/Qwen2.5-3B-bnb-4bit`
- **Dataset**: `5CD-AI/Vietnamese-alpaca-gpt4-gg-translated`, 193 cleaned samples (173 train + 20 eval)
- **max_seq_length**: 1024 (p95 = 562, rounded up to the next power of 2 and capped at 1024)
- **GPU**: Kaggle Tesla T4, ~15 GB VRAM
- **Training cost**: $0.13 (~22.7 phút @ $0.35/hr)
- **HF Hub link**: https://huggingface.co/nxhai/lab21_qwen25_3b_vi_t4-r64
- **GitHub repo**: https://github.com/Nxhaicr7/Day21-Track3-Finetuning-LLMs-LoRA-QLoRA.git

## 2. Rank Experiment Results

| Variant | Rank | Target Scope | DoRA | Trainable Params | Train Time | Peak VRAM | Eval Loss | Perplexity |
|---------|------|--------------|------|------------------|------------|-----------|-----------|------------|
| base | - | base model | no | 0 | - | - | 1.9403 | 6.9606 |
| r8 | 8 | q+v | no | 1,843,200 | 5.42 min | 8.47 GB | 1.5568 | 4.7434 |
| r16 | 16 | q+v | no | 3,686,400 | 5.47 min | 7.87 GB | 1.5088 | 4.5212 |
| r64 | 64 | q+v | no | 14,745,600 | 5.43 min | 9.25 GB | 1.4461 | 4.2466 |
| r16_all_layers | 16 | all layers | no | 29,933,568 | 6.33 min | 10.00 GB | 1.4533 | 4.2771 |

Nhận xét nhanh:
- **Perplexity tốt nhất** là `r64` với `4.2466`, giảm rõ rệt so với base `6.9606`.
- **ROI tốt nhất** theo mình là `r64`: thời gian train gần như không tăng so với `r8` và `r16`, nhưng perplexity cải thiện rõ nhất.
- **Stretch goal đáng giá nhất** là `r16_all_layers`: dù chưa vượt `r64`, nó cho perplexity `4.2771`, khá sát `r64`, chứng minh việc mở rộng `target_modules` là một hướng tối ưu hợp lý.

## 3. Loss Curve Analysis
Mình đã lưu `results/loss_curve.png` để đính kèm khi nộp. Trong quá trình train baseline `r16`, train loss giảm đều và không có dấu hiệu dao động bất thường, cho thấy pipeline QLoRA + gradient checkpointing hoạt động ổn định trên T4.

Do notebook tắt `eval_strategy` trong lúc train để tránh OOM trên T4, mình không có eval curve theo từng checkpoint. Vì vậy, mình không thể kết luận overfitting theo nghĩa kinh điển là “train loss giảm trong khi eval loss tăng” qua thời gian. Tuy nhiên, eval perplexity cuối cùng của các rank đều thấp hơn khá nhiều so với base model, nên có thể xem đây là tín hiệu rằng fine-tuning đã giúp mô hình thích nghi với dataset thay vì chỉ học thuộc hoàn toàn. Với dataset nhỏ 193 mẫu, nguy cơ overfitting vẫn tồn tại, nhưng từ loss curve mượt và kết quả eval cuối cùng thì mình đánh giá mô hình vẫn đang ở trạng thái chấp nhận được cho quy mô lab.

## 4. Qualitative Comparison (5 examples)

Lưu ý: phần qualitative CSV hiện được generate với adapter `r16`, đúng với tinh thần rubric là so sánh base vs fine-tuned baseline. Trong khi đó, biến thể có perplexity tốt nhất toàn cục là `r64`.

### Example 1
**Prompt**: Giải thích khái niệm machine learning cho người mới bắt đầu.  
**Base**: Giải thích đúng ý chính, ngắn gọn và khá tự nhiên.  
**Fine-tuned (r16)**: Trả lời dài hơn nhưng lặp ý nhiều, có xu hướng diễn đạt vòng lặp.  
**Nhận xét**: Case này fine-tuned chưa thắng rõ base; đây là một ví dụ cho thấy perplexity tốt hơn chưa chắc luôn tương ứng với diễn đạt ngắn gọn hơn.

### Example 2
**Prompt**: Viết đoạn code Python tính số Fibonacci thứ n.  
**Base**: Trả lời trực tiếp bằng code đệ quy, đúng nhưng tối giản.  
**Fine-tuned (r16)**: Trả lời có giải thích và đưa ví dụ code dùng vòng lặp, hữu ích hơn cho người học.  
**Nhận xét**: Fine-tuned tốt hơn base về tính hướng dẫn và tính sư phạm.

### Example 3
**Prompt**: Liệt kê 5 nguyên tắc thiết kế UI/UX.  
**Base**: Có dấu hiệu lỗi định dạng và xuất hiện ký tự lạ, câu trả lời bị nhiễu.  
**Fine-tuned (r16)**: Cấu trúc rõ hơn, liệt kê mạch lạc hơn dù vẫn còn hơi chung chung.  
**Nhận xét**: Đây là ví dụ fine-tuned cải thiện độ sạch định dạng và khả năng giữ cấu trúc đầu ra.

### Example 4
**Prompt**: Tóm tắt sự khác biệt giữa LoRA và QLoRA.  
**Base**: Trả lời đúng ý tổng quát nhưng còn đơn giản, giải thích chưa thật sắc nét.  
**Fine-tuned (r16)**: Dùng cách giải thích dài hơn, có xu hướng mang tính “giảng giải”, làm rõ hơn khác biệt giữa adapter tuning và quantization.  
**Nhận xét**: Fine-tuned nhỉnh hơn về độ đầy đủ nội dung, dù có thể rút gọn thêm cho súc tích.

### Example 5
**Prompt**: Phân biệt prompt engineering, RAG, và fine-tuning.  
**Base**: Trả lời đúng hướng nhưng bị cắt ngắn và chưa tách bạch ba khái niệm rõ ràng.  
**Fine-tuned (r16)**: Câu trả lời có cấu trúc hơn, mô tả được ba kỹ thuật như ba cách tiếp cận khác nhau để cải thiện hiệu suất mô hình.  
**Nhận xét**: Fine-tuned tốt hơn về tính phân loại khái niệm và độ dễ đọc.

Tổng hợp qualitative:
- Fine-tuned thắng rõ ở các prompt cần cấu trúc, giải thích từng bước, hoặc phân loại khái niệm.
- Base đôi khi trả lời ngắn gọn hơn và ít lặp hơn, đặc biệt ở prompt giải thích cơ bản.
- Điều này phù hợp với bản chất fine-tuning trên dataset instruction-following: mô hình học được phong cách trả lời có cấu trúc hơn, nhưng đôi lúc đánh đổi bằng độ dài và sự lặp lại.

## 5. Conclusion về Rank Trade-off
Trong thí nghiệm này, `r64` là lựa chọn tốt nhất nếu chỉ xét chất lượng cuối cùng trên eval set, vì nó đạt perplexity thấp nhất `4.2466`, tốt hơn cả `r16` (`4.5212`) và `r8` (`4.7434`). Điều thú vị là thời gian train của `r64` gần như không khác nhiều so với `r8` và `r16` trên cùng cấu hình T4, nên chi phí thời gian tăng không đáng kể. Đổi lại, `r64` tốn VRAM hơn, khoảng `9.25 GB`, vẫn nằm trong khả năng của T4 nhưng cao hơn baseline. Mình xem đây là một ví dụ rõ về trade-off: tăng rank giúp cải thiện năng lực biểu diễn adapter, và ở dataset này thì lợi ích đó vẫn còn thấy rõ từ `r16` lên `r64`, nên chưa xuất hiện diminishing returns mạnh ở mốc đó.

Tuy nhiên, khi xét stretch goal `r16_all_layers`, perplexity đạt `4.2771`, khá sát `r64`, nhưng số trainable parameters tăng lên gần gấp đôi `r64` và peak VRAM cũng tăng lên `10.00 GB`. Điều này cho thấy target all layers là một hướng mạnh, nhưng chưa chắc có ROI tốt nhất trong bối cảnh GPU hạn chế. Nếu triển khai production cho bài toán tương tự, mình sẽ chọn `r64` vì nó cho chất lượng tốt nhất với chi phí tăng thêm vừa phải. Nếu cần một cấu hình bảo thủ hơn về VRAM hoặc muốn giữ sát lab spec, `r16` vẫn là lựa chọn cân bằng. Bonus DoRA không chạy ổn định trên runtime này do lỗi mixed dtype giữa Q/K/V, nên mình không xem nó là lựa chọn thực tiễn trong môi trường T4 hiện tại.

## 6. What I Learned
- Rank cao hơn không phải lúc nào cũng tốn thêm quá nhiều thời gian train, nhưng có thể cải thiện perplexity đáng kể nếu dataset đủ tín hiệu.
- Việc chọn `target_modules` ảnh hưởng rất mạnh đến số trainable parameters và VRAM, nên “target all layers” cần được cân nhắc bằng cả chất lượng lẫn tài nguyên.
- Trên GPU nhỏ như T4, thiết kế pipeline an toàn như `packing=False`, `safe_evaluate()`, gradient checkpointing và save adapter trước eval là cực kỳ quan trọng để hoàn thành lab trơn tru.

## Optional Bonus Notes
- **Stretch goal thành công**: `target ALL layers` với `r16_all_layers`
- **Stretch goal chưa ổn định**: `DoRA` fail trên runtime T4 + Unsloth + xformers vì mixed dtype Q/K/V
- **W&B run link**: không sử dụng trong lần chạy này
- **GGUF / merged artifact**: chưa thực hiện
- **Best submission strategy**: Option B vì adapter đã được push công khai lên Hugging Face Hub và có thể verify trực tiếp.

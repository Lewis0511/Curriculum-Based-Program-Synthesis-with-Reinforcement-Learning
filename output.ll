; ModuleID = "module"
target triple = "unknown-unknown-unknown"
target datalayout = ""

define void @"main"(i32* %"arr")
{
entry:
  %"ptr_0" = getelementptr i32, i32* %"arr", i32 1
  %"int_0" = load i32, i32* %"ptr_0"
  %"int_1" = load i32, i32* %"ptr_0"
  %"bool_0" = icmp sgt i32 %"int_0", %"int_1"
  store i32 %"int_1", i32* %"ptr_0"
  %"bool_1" = icmp slt i32 %"int_1", %"int_0"
  %"ptr_1" = getelementptr i32, i32* %"arr", i32 2
  %"int_2" = load i32, i32* %"ptr_1"
  %"ptr_2" = getelementptr i32, i32* %"arr", i32 2
  %"ptr_3" = getelementptr i32, i32* %"arr", i32 1
  ret void
}

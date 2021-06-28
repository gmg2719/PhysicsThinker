// MIT License
// 
// Copyright (c) 2021 PingzhouMing
// 
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
// 
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
// 
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
#include <iostream>
#include "my_logger.hpp"

using namespace std;

int main(void)
{
    std::cout << "This is the world !" << std::endl;

    AixLog::Log::init<AixLog::SinkCout>(AixLog::Severity::trace);
    LOG(TRACE, "LOG_TAG") << "TEST 1\n";
    LOG(DEBUG, "LOG_TAG") << "TEST2\n";
    LOG(INFO, "LOG_TAG") << "TEST3\n";

    AixLog::Log::init({/// Log everything into file "all.log"
                       make_shared<AixLog::SinkFile>(AixLog::Severity::trace, "all.log"),
                       /// Log everything to SinkCout
                       make_shared<AixLog::SinkCout>(AixLog::Severity::trace, "cout: %Y-%m-%d %H-%M-%S.#ms [#severity] (#tag_func) #message"),
                       /// Log error and higher severity messages to cerr
                       make_shared<AixLog::SinkCerr>(AixLog::Severity::error, "cerr: %Y-%m-%d %H-%M-%S.#ms [#severity] (#tag_func)"),
                       /// Callback log sink with cout logging in a lambda function
                       /// Could also do file logging
                       make_shared<AixLog::SinkCallback>(AixLog::Severity::trace, [](const AixLog::Metadata& metadata, const std::string& message) {
                           cout << "Callback:\n\tmsg:   " << message << "\n\ttag:   " << metadata.tag.text
                                << "\n\tsever: " << AixLog::to_string(metadata.severity) << " (" << static_cast<int>(metadata.severity) << ")\n";
                           if (metadata.timestamp)
                               cout << "\ttime:  " << metadata.timestamp.to_string() << "\n";
                           if (metadata.function)
                               cout << "\tfunc:  " << metadata.function.name << "\n\tline:  " << metadata.function.line
                                    << "\n\tfile:  " << metadata.function.file << "\n";
                       })});
    /// Log with info severity
    LOG(INFO) << "LOG(INFO)\n";
    /// ... with a tag
    LOG(INFO, "guten tag") << "LOG(INFO, \"guten tag\")\n";
    /// ... with an explicit tag (same result as above)
    LOG(INFO) << TAG("guten tag") << "LOG(INFO) << TAG(\"guten tag\")\n";

    /// Different log severities
    LOG(FATAL) << "LOG(FATAL)\nLOG(FATAL) Second line\n";
    LOG(FATAL) << TAG("hello") << "LOG(FATAL) << TAG(\"hello\") no line break";
    LOG(FATAL) << "LOG(FATAL) 2 no line break";
    LOG(ERROR) << "LOG(ERROR): change in log-level will add a line break";
    LOG(WARNING) << "LOG(WARNING)";
    LOG(NOTICE) << "LOG(NOTICE)";
    LOG(INFO) << "LOG(INFO)\n";
    LOG(INFO) << TAG("my tag") << "LOG(INFO) << TAG(\"my tag\")\n";
    LOG(DEBUG) << "LOG(DEBUG)\n";
    LOG(TRACE) << "LOG(TRACE)\n";

    /// Conditional logging
    LOG(DEBUG) << COND(1 == 1) << "LOG(DEBUG) will be logged\n";
    LOG(DEBUG) << COND(1 == 2) << "LOG(DEBUG) will not be logged\n";

    /// Colors :-)
    LOG(FATAL) << "LOG(FATAL) " << AixLog::Color::red << "red" << AixLog::Color::none << ", default color\n";
    LOG(FATAL) << "LOG(FATAL) " << COLOR(red) << "red" << COLOR(none) << ", default color (using macros)\n";
    LOG(FATAL) << "LOG(FATAL) " << AixLog::TextColor(AixLog::Color::yellow, AixLog::Color::blue) << "yellow on blue background" << AixLog::Color::none
               << ", default color\n";

    AixLog::Severity severity(AixLog::Severity::debug);
    LOG(severity) << "LOG(severity) << severity\n";

    return 0;
}

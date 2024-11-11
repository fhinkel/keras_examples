#include <cstdio>
#include <iostream>
#include <atomic>
#include <thread>

std::atomic<int> counter(0);
std::mutex mtx;
int counter2 = 0;

void increment()
{
    for (int i = 0; i < 1000; ++i)
    {
        std::unique_lock lck {mtx}; // Acquires the mutex and auto releases when out of scope
        counter2++;
    }

    for (int i = 0; i < 1000; ++i)
    {
        counter.fetch_add(1, std::memory_order_relaxed);
    }
}

int main()
{
    std::thread t1(increment);
    std::thread t2(increment);

    t1.join();
    t2.join();

    std::cout << "Counter: " << counter << std::endl;
    std::cout << "Unique Counter2: " << counter2 << std::endl;
    return 0;
}
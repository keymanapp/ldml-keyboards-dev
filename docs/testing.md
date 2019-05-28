# Testing

The program `kbdtest` runs a battery of tests, represented in a `.xml` file against an LDML keyboard. An example test file is:

```
<?xml version="1.0" encoding="utf-8"?>
<ldmlkeytests>
    <test-sequence id="1">
            <event key="[C01]" output="a"/>
            <event key="[shift C01]" output="aA"/>
    </test-sequence>
</ldmlkeytests>
```

There may be multiple `test-sequence` elements each with its own `id` attribute. A test sequence consists of a sequence of `event` elements that specify the `key` sequence (usually just one key) to be pressed and the expected output after that `key` sequence has been typed. Notice that events build on each other. So in our example, the output for the second event takes into account the output of the first event as its input context along with the key pressed. Keys are surrounded by `[]` to allow for multiple key strokes in a single event (e.g. for deadkeys).

## Test Types

A test sequence undergoes a number of tests.

### Basic

The most basic test is that given the context of the output of the previous event, the key(s) in the `key` attribute are pressed and the resulting output is tested against the expected `output` attribute. This is the first test and is always executed since it provides a context for all the other tests.

### Backspace

The backspace test tests the effect of pressing backspace. This is not done through undo, but exercises the backspace transforms and default backspace handling of the LDML engine. The test walks backwards through all the events. It presses backspace for each event it wants to go back and then retypes the keys for those events to try to recreate the current output.

For example, in our test file, when testing the second event, the backspace test will first press backspace once and then retype the `[Shift C01]` and check the output. Then it will press backspace twice and retype `[C01] [Shift C01]` and check the output.

There are times when the backspace handling does not reflect undoing keystrokes. For example:

```
<?xml version="1.0" encoding="utf-8"?>
<ldmlkeytests>
    <test-sequence id="1">
        <event key="[B05]" output="&#x1C13;"/>
        <event key="[C09]" output="&#x1C14;" skip="bksp"/>
    </test-sequence>
</ldmlkeytests>
```

Here the first character is transformed by a subsequent keystroke into another character. It is unlikely that if someone drops a cursor in some text after a U+1C14, that on pressing backspace, they would want that character to turn into a U+1C13 rather than simply being deleted. And that is how this keyboard is programmed, to delete the U+1C14. But this breaks the backspace test since pressing backspace and then `[C09]` will not result in U+1C14.

The backspace test may be skipped if the `skip` attribute contains in its list of words the word `bksp`. This also has the effect of blocking later tests from trying to backspace past this event as it walks back through the events from a later one.

The default behaviour of the engine for backspace is to remove the last character. In terms of normalization a character is a single NFC codepoint.

### Normal Test

The normalization test tests that the output of an event, if normalized, is unchanged. It does this for both NFC and NFD. If for a normal form, the output of an event is not the same when normalized into a form, then a further test is done. The keystroke is undone (as opposed to hitting backspace) and the previous output is normalized. This is fed in as the context for the keystrokes for this event and the output compared against the normalized form of the test output.

Notice that the engine runs in NFD and returns results in NFD.

This test can be skipped by adding the word `normal` to the `skip` attribute. The test is most often skipped where the keyboard is expected to reorder a diacritic to after a following base (in a separate event). When there is no base character in the output, this can cause problems and is not really a fair test of a keyboard with regard to normalization.
